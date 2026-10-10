"""MFCL image/text features fused with a typed CHECKED comment graph."""

from __future__ import annotations

import torch
import torch.nn as nn

from comment_graph.dynamic_graph import (
    GRAPH_STRUCTURES,
    POOLING_MODES,
    CrossModalQualityGate,
    DualModalCommentAttention,
    DynamicCrossModalEncoder,
    SourceAwareQualityAttention,
    parent_semantic_consistency_gate,
    quality_weighted_pool,
)
from Weibo.online_encoder import OnlineFeatureEncoder
from Weibo.propogation import ObjectiveLoss_Graph, ObjectiveLoss_TextImage
from comment_graph.local_correction import DECISION_MODES, CommentLocalCorrection
from comment_graph.isolated_fusion import GeneratedGraphCorrection, select_nodes
from comment_graph.mts_fusion import MTSFusion
from comment_graph.mimoe_fusion import MIMoEFusion
from comment_graph.evidence_fusion import EVIDENCE_MODES, EntityEvidenceFusion
from comment_graph.typed_graph import TypedCommentEncoder
from comment_graph.generated_reliability import RELIABILITY_MODES, refine_generated_graph
from comment_graph.reliability_residual import ReliabilityResidual


class CommentGraphMFCL(nn.Module):
    """Use MFCL encoders/GNN and concatenate multimodal and graph vectors."""

    def __init__(
        self,
        config,
        mode: str,
        graph_structure: str | None = None,
        pooling_mode: str | None = None,
        decision_mode: str | None = None,
    ):
        super().__init__()
        self.mode = mode
        self.mm_fusion_mode = str(config.get("mm_fusion", {}).get("mode", "mfcl"))
        if self.mm_fusion_mode not in {"mfcl", "mts", "mimoe"}:
            raise ValueError("Unknown multimodal fusion mode")
        self.evidence_mode = str(config.get("entity_evidence", {}).get("mode", "none"))
        if self.evidence_mode not in EVIDENCE_MODES:
            raise ValueError(f"Unknown entity evidence mode: {self.evidence_mode}")
        self.decision_mode = decision_mode or config.get("decision_mode", "concat")
        if self.decision_mode not in DECISION_MODES:
            raise ValueError(f"Unknown decision mode: {self.decision_mode}")
        self.graph_structure = graph_structure or str(config.graph.structure_mode)
        self.pooling_mode = pooling_mode or str(config.graph.pooling_mode)
        self.graph_message_passing = str(config.graph.get("message_passing", "legacy_gcn"))
        self.semantic_parent_gate = bool(config.graph.get("semantic_parent_gate", False))
        self.semantic_parent_gate_floor = float(config.graph.get("semantic_parent_gate_floor", 0.25))
        reliability_options = config.graph.get("generated_reliability", {})
        self.generated_reliability_mode = str(reliability_options.get("mode", "none"))
        self.generated_reliability_options = dict(reliability_options)
        if self.generated_reliability_mode not in RELIABILITY_MODES:
            raise ValueError("Unknown generated reliability mode")
        if self.generated_reliability_mode != "none":
            if self.graph_message_passing != "typed_rgcn" or self.graph_structure != "observed_tree":
                raise ValueError("Generated reliability requires typed_rgcn with observed_tree")
            if self.pooling_mode != "mean" or self.semantic_parent_gate:
                raise ValueError("Generated reliability requires mean pooling without the legacy parent gate")
            if self.decision_mode == "source_isolated":
                raise ValueError("Generated reliability is not implemented for source_isolated fusion")
        if self.graph_structure not in GRAPH_STRUCTURES:
            raise ValueError(f"Unknown graph structure: {self.graph_structure}")
        if self.pooling_mode not in POOLING_MODES:
            raise ValueError(f"Unknown pooling mode: {self.pooling_mode}")
        if self.graph_message_passing not in {"legacy_gcn", "typed_rgcn"}:
            raise ValueError("graph.message_passing must be legacy_gcn or typed_rgcn")
        if self.graph_message_passing == "typed_rgcn" and self.graph_structure == "dynamic_cross_modal":
            raise ValueError("typed_rgcn requires supplied relation edges, not dynamic_cross_modal")
        if self.decision_mode == "source_isolated" and self.pooling_mode != "mean":
            raise ValueError("source_isolated currently requires --pooling-mode mean")
        if self.decision_mode == "reliability_residual" and self.mode != "real_generated":
            raise ValueError("reliability_residual requires real_generated mode")
        if self.decision_mode == "reliability_residual" and (
                self.mm_fusion_mode != "mfcl" or self.evidence_mode != "none"):
            raise ValueError("reliability_residual currently requires MFCL fusion without entity evidence")
        output_dim = int(config.encoder.output_dim)
        self.feature_encoder = OnlineFeatureEncoder(
            text_model_name=str(config.encoder.text_model_name),
            output_dim=output_dim,
            pretrained_image=bool(config.encoder.pretrained_image),
            fine_tune_text=bool(config.encoder.fine_tune_text),
            fine_tune_image=bool(config.encoder.fine_tune_image),
            node_encode_batch_size=int(config.encoder.node_encode_batch_size),
        )
        self.text_image_model = ObjectiveLoss_TextImage(config)
        self.graph_model = ObjectiveLoss_Graph(
            num_hidden=int(config.graph.hidden_dim),
            num_gc_layers=int(config.graph.num_layers) + 1,
        )
        self.typed_graph_encoder = (
            TypedCommentEncoder(
                input_dim=output_dim,
                hidden_dim=int(config.graph.hidden_dim),
                num_relations=int(config.graph.get("num_edge_relations", 2)),
            )
            if self.graph_message_passing == "typed_rgcn"
            else None
        )
        self.node_type_embedding = nn.Embedding(int(config.graph.node_type_count), output_dim)

        self.h_mm_dim = (int(config.get("mm_fusion", {}).get("hidden_dim", 768))
                         if self.mm_fusion_mode == "mts" else output_dim * 2)
        # Existing two-layer MFCL GCN pools 512 + 256 dimensions.
        self.h_graph_dim = int(config.graph.hidden_dim) * 3
        self.dynamic_encoder = DynamicCrossModalEncoder(
            self.graph_model.encoder.convs,
            top_k=int(config.graph.dynamic_top_k),
            score_dim=int(config.graph.edge_score_dim),
        )
        self.comment_attention = DualModalCommentAttention(
            self.h_graph_dim,
            dropout=float(config.graph.attention_dropout),
        )
        self.quality_gate = CrossModalQualityGate(
            node_dim=output_dim,
            query_dim=output_dim,
            score_dim=int(config.graph.quality_gate_dim),
        )
        self.quality_attention = SourceAwareQualityAttention(
            node_dim=self.h_graph_dim,
            query_dim=output_dim,
            score_dim=int(config.graph.attention_score_dim),
            dropout=float(config.graph.quality_attention_dropout),
            real_residual_init=float(config.graph.quality_real_residual_init),
            generated_residual_init=float(config.graph.quality_generated_residual_init),
        )
        self.graph_attention_norm = nn.LayerNorm(self.h_graph_dim)
        self.h_final_dim = self.h_mm_dim + self.h_graph_dim
        self.num_classes = int(config.get("classification", {}).get("num_classes", 2))
        if self.num_classes < 2:
            raise ValueError("classification.num_classes must be at least 2")
        self.classifier = nn.Sequential(
            nn.Linear(self.h_final_dim, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, self.num_classes),
        )
        self._init_new_parameters()
        if self.mm_fusion_mode == "mts":
            # Initialization is local to this optional module, preserving RNG
            # consumption for downstream modules shared with the control.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(config.seed) + 50000)
                self.mts_fusion = MTSFusion(output_dim, self.h_mm_dim,
                                            int(config.get("mm_fusion", {}).get("layers", 8)))
        elif self.mm_fusion_mode == "mimoe":
            options = config.get("mimoe", {})
            # Build after the common classifier so this optional arm cannot
            # alter the initialization of the paired MFCL control.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(config.seed) + 60000)
                self.mimoe_fusion = MIMoEFusion(
                    input_dim=output_dim,
                    semantic_dim=int(options.get("semantic_dim", 64)),
                    hidden_dim=int(options.get("hidden_dim", output_dim)),
                    num_experts=int(options.get("num_experts", 2)),
                    dropout=float(options.get("dropout", .1)),
                    agreement_threshold=float(options.get("agreement_threshold", .3)),
                    semantic_threshold=float(options.get("semantic_threshold", .3)),
                    interaction_loss_weight=float(options.get("interaction_loss_weight", .7)),
                    balance_loss_weight=float(options.get("balance_loss_weight", .1)),
                    router_z_loss_weight=float(options.get("router_z_loss_weight", .01)),
                )
        if self.evidence_mode != "none":
            options = config.get("entity_evidence", {})
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(config.seed) + 70000)
                self.entity_evidence_fusion = EntityEvidenceFusion(
                    self.h_mm_dim, int(options.get("hidden_dim", 256)),
                    float(options.get("dropout", .1)), float(options.get("gate_init", .1)),
                    self.evidence_mode,
                )
        # Instantiate only when enabled, preserving old initialization/state dicts.
        if self.decision_mode == "global_local_correction":
            options = config.get("local_correction", {})
            self.local_correction = CommentLocalCorrection(
                global_dim=self.h_final_dim,
                num_classes=self.num_classes,
                dim=int(options.get("dim", 128)),
                temperature=float(options.get("temperature", 1.0)),
                dropout=float(options.get("dropout", 0.1)),
                gate_init=float(options.get("gate_init", 0.1)),
            )
        elif self.decision_mode == "source_isolated":
            self.generated_branch = GeneratedGraphCorrection(config, self.h_final_dim)
        elif self.decision_mode == "reliability_residual":
            options = config.get("reliability_residual", {})
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(config.seed) + 80000)
                self.reliability_residual = ReliabilityResidual(
                    self.h_graph_dim, self.h_final_dim, self.num_classes,
                    hidden_dim=int(options.get("hidden_dim", 128)),
                    gate_init=float(options.get("gate_init", 0.2)),
                    max_logit_delta=float(options.get("max_logit_delta", 2.0)),
                    branch_dropout=float(options.get("branch_dropout", 0.1)),
                )

    def _init_new_parameters(self) -> None:
        nn.init.normal_(self.node_type_embedding.weight, mean=0.0, std=0.02)
        for module in (
            self.dynamic_encoder.edge_builders,
            self.comment_attention,
            self.quality_gate,
            self.quality_attention,
        ):
            for child in module.modules():
                if isinstance(child, nn.Linear):
                    nn.init.xavier_uniform_(child.weight)
                    if child.bias is not None:
                        nn.init.zeros_(child.bias)
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, batch_data, disable_generated=False, return_generated_context=False):
        if self.decision_mode == "source_isolated":
            return self._forward_isolated(batch_data, disable_generated, return_generated_context)
        if self.decision_mode == "reliability_residual":
            if disable_generated or return_generated_context:
                raise ValueError("Use a real-only graph view to audit reliability_residual")
            return self._forward_reliability_residual(batch_data)
        if return_generated_context:
            raise ValueError("Generated context is only available for source_isolated")
        if disable_generated:
            raise ValueError("disable_generated is only supported by source_isolated")
        return self._forward_standard(batch_data)

    def _forward_reliability_residual(self, batch_data):
        real_view, real_indices = select_nodes(batch_data, batch_data.node_type != 2)
        base = self._forward_standard(real_view)
        node_features = base["node_features"].new_zeros((batch_data.num_nodes, base["node_features"].size(-1)))
        node_features[real_indices] = base["node_features"]
        generated_indices = torch.nonzero(batch_data.node_type == 2, as_tuple=False).flatten()
        if generated_indices.numel():
            chunks = []
            for indices in generated_indices.split(self.feature_encoder.node_encode_batch_size):
                tokens = self.feature_encoder._text_forward(
                    batch_data.node_input_ids[indices], batch_data.node_attention_mask[indices],
                )
                chunks.append(self.feature_encoder._masked_mean(tokens, batch_data.node_attention_mask[indices]))
            node_features[generated_indices] = self.feature_encoder.text_projection(torch.cat(chunks))
        # Reuse exactly the base image/news representations and their BN pass.
        # The full graph supplies evidence only, not a second classifier pass.
        full = self._forward_standard(
            batch_data, compute_logits=False,
            precomputed=(base["root_feature"], base["image_feature_map"], node_features,
                         base["text_embedding"], base["image_embedding"], base["h_mm"]),
        )
        correction = self.reliability_residual(
            base["logits"], base["h_final"], base["h_graph"], full["h_graph"],
            full["node_quality"], batch_data.node_type, batch_data.batch,
        )
        full["logits"] = correction["logits"]
        full["global_logits"] = base["logits"]
        full["local_correction"] = correction
        full["h_real_graph"] = base["h_graph"]
        return full

    def _forward_isolated(self, batch_data, disable_generated, return_generated_context=False):
        keep = batch_data.node_type == 0
        if self.mode in {"real_only", "real_generated"}:
            keep = keep | (batch_data.node_type == 1)
        base_view, base_indices = select_nodes(batch_data, keep)
        # Encode the base view FIRST. Generated text cannot even change its
        # encoder chunk boundaries or its forward-pass random-number usage.
        output = self._forward_standard(base_view)
        for key in ("node_text_attention", "node_image_attention", "node_quality", "node_source_attention"):
            # No per-node quality estimator is used in this mode. Preserve a
            # neutral quality=1 instead of misreporting synthetic nodes as bad.
            values = (output[key].new_ones(batch_data.num_nodes) if key == "node_quality"
                      else output[key].new_zeros(batch_data.num_nodes))
            values[base_indices] = output[key]
            output[key] = values
        output["base_node_indices"] = base_indices
        output["base_edge_history"] = output["edge_history"]
        zero_logits = output["logits"].float().new_zeros(output["logits"].shape)
        correction = {"logit_delta": zero_logits, "local_logits": zero_logits,
                      "correction_gate": zero_logits[:, :1], "branch_active": zero_logits[:, :1],
                      "h_generated_graph": output["h_graph"].new_zeros(output["h_graph"].shape),
                      "edge_history": []}
        allowed = self.mode in {"generated_only", "real_generated"}
        if allowed and not disable_generated and (batch_data.node_type == 2).any():
            view, indices = select_nodes(batch_data, batch_data.node_type != 1)
            chunks = []
            # Same BERT and projection as real comments; do not re-encode images.
            for start in range(0, view.num_nodes, self.feature_encoder.node_encode_batch_size):
                end = start + self.feature_encoder.node_encode_batch_size
                mask = view.node_attention_mask[start:end]
                tokens = self.feature_encoder._text_forward(view.node_input_ids[start:end], mask)
                chunks.append(self.feature_encoder._masked_mean(tokens, mask))
            features = self.feature_encoder.text_projection(torch.cat(chunks))
            with torch.autocast(device_type=features.device.type, enabled=False):
                correction = self.generated_branch(
                    features.float(), view, output["h_final"].float(),
                    output["text_embedding"].float(), output["image_embedding"].float(),
                    self.graph_structure,
                )
            output["generated_node_indices"] = indices
            if return_generated_context:
                output["generated_context"] = {
                    "features": features, "view": view,
                    "base_features": output["h_final"],
                    "text": output["text_embedding"], "image": output["image_embedding"],
                }
        else:
            output["generated_node_indices"] = base_indices.new_empty(0)
        output["local_correction"] = correction
        output["logits"] = output["global_logits"].float() + correction["logit_delta"]
        return output

    def _forward_standard(self, batch_data, compute_logits=True, precomputed=None):
        if precomputed is None:
            root_feature, image_feature_map, node_features = self.feature_encoder(batch_data)
        else:
            root_feature, image_feature_map, node_features = precomputed[:3]
        device_type = root_feature.device.type
        # MFCL's original attention mask uses -1e18. That value cannot be
        # represented by FP16 and raises an overflow error under autocast.
        # Keep the reused legacy attention/GNN path in FP32 without changing
        # the original baseline implementation. The surrounding encoder and
        # classifier may still use AMP.
        with torch.autocast(device_type=device_type, enabled=False):
            batch_data.root_feature = root_feature.float()
            batch_data.img = image_feature_map.float()

            fusion_output = {}
            if precomputed is not None:
                text_embedding, image_embedding, h_mm = precomputed[3:]
            else:
                text_embedding, image_embedding = self.text_image_model(batch_data)
                if self.mm_fusion_mode == "mts":
                    h_mm = self.mts_fusion(text_embedding, image_embedding)
                elif self.mm_fusion_mode == "mimoe":
                    fusion_output = self.mimoe_fusion(text_embedding, image_embedding)
                    h_mm = fusion_output["h_mm"]
                else:
                    h_mm = torch.cat((text_embedding, image_embedding), dim=-1)
            evidence_output = {}
            if precomputed is None and self.evidence_mode != "none":
                required = ("entity_evidence_scores", "entity_evidence_text_sims", "entity_evidence_image_sims",
                            "entity_evidence_reliability", "entity_evidence_mask")
                if any(not hasattr(batch_data, key) for key in required):
                    raise ValueError("Entity evidence is enabled but the batch has no label-free evidence fields")
                evidence_output = self.entity_evidence_fusion(
                    h_mm, batch_data.entity_evidence_scores, batch_data.entity_evidence_text_sims,
                    batch_data.entity_evidence_image_sims, batch_data.entity_evidence_reliability,
                    batch_data.entity_evidence_mask,
                )
                h_mm = evidence_output["h_mm"]

            if self.mode == "baseline":
                h_graph = h_mm.new_zeros((h_mm.size(0), self.h_graph_dim))
                node_text_attention = h_mm.new_zeros((batch_data.num_nodes,))
                node_image_attention = h_mm.new_zeros((batch_data.num_nodes,))
                node_quality = h_mm.new_ones((batch_data.num_nodes,))
                semantic_parent_gate = h_mm.new_ones((batch_data.num_nodes,))
                node_source_attention = h_mm.new_zeros((batch_data.num_nodes,))
                quality_alpha_real = h_mm.new_zeros(())
                quality_alpha_generated = h_mm.new_zeros(())
                edge_history = []
                generated_reliability_diagnostics = []
            else:
                typed_nodes = node_features.float() + self.node_type_embedding(batch_data.node_type)
                graph_edge_index, graph_edge_type = batch_data.edge_index, batch_data.edge_type
                generated_reliability_diagnostics = []
                if self.semantic_parent_gate:
                    semantic_parent_gate = parent_semantic_consistency_gate(
                        node_features.float(), batch_data.node_type, batch_data.edge_index,
                        batch_data.edge_type, self.semantic_parent_gate_floor,
                    )
                    typed_nodes = typed_nodes * semantic_parent_gate.unsqueeze(-1)
                else:
                    semantic_parent_gate = typed_nodes.new_ones((typed_nodes.size(0),))
                if self.pooling_mode == "quality_gated_attention":
                    node_quality = self.quality_gate(
                        typed_nodes,
                        batch_data.node_type,
                        batch_data.batch,
                        text_embedding,
                        image_embedding,
                    )
                    graph_input_nodes = typed_nodes * node_quality.unsqueeze(-1)
                else:
                    node_quality = typed_nodes.new_ones((typed_nodes.size(0),))
                    graph_input_nodes = typed_nodes
                if self.generated_reliability_mode != "none":
                    opts = self.generated_reliability_options
                    graph_edge_index, graph_edge_type, text_reliability, generated_reliability_diagnostics = (
                        refine_generated_graph(
                            node_features.float(), batch_data.node_type,
                            graph_edge_index, graph_edge_type, batch_data.batch,
                            self.generated_reliability_mode,
                            predicted_parent_bonus=float(opts.get("predicted_parent_bonus", 0.10)),
                            min_parent_similarity=float(opts.get("min_parent_similarity", 0.45)),
                            relevance_floor=float(opts.get("relevance_floor", 0.20)),
                            relevance_low=float(opts.get("relevance_low", 0.35)),
                            relevance_high=float(opts.get("relevance_high", 0.70)),
                            duplicate_start=float(opts.get("duplicate_start", 0.90)),
                            max_generated_depth=int(opts.get("max_generated_depth", 3)),
                            confidence_floor=float(opts.get("confidence_floor", 0.25)),
                            root_fallback_percentile=float(opts.get("root_fallback_percentile", 0.05)),
                            anchor_low=float(opts.get("anchor_low", 0.75)),
                            anchor_high=float(opts.get("anchor_high", 0.84)),
                            anchor_floor=float(opts.get("anchor_floor", 0.20)),
                        )
                    )
                    if self.generated_reliability_mode in {"quality", "both", "quality_confidence",
                                                           "ancestry_anchor"}:
                        visual_quality = self.quality_gate(
                            typed_nodes, batch_data.node_type, batch_data.batch,
                            text_embedding, image_embedding,
                        )
                        node_quality = text_reliability.clone()
                        generated_mask = batch_data.node_type == 2
                        node_quality[generated_mask] = (
                            text_reliability[generated_mask] *
                            (0.5 + 0.5 * visual_quality[generated_mask])
                        )
                        for item in generated_reliability_diagnostics:
                            item["fused_reliability"] = float(node_quality[item["node_index"]].detach())
                    else:
                        node_quality = text_reliability
                if self.graph_structure == "dynamic_cross_modal":
                    h_graph, graph_nodes, edge_history = self.dynamic_encoder(
                        graph_input_nodes,
                        batch_data.node_type,
                        batch_data.batch,
                        text_embedding,
                        image_embedding,
                        node_quality=(
                            node_quality
                            if self.pooling_mode == "quality_gated_attention"
                            else None
                        ),
                    )
                else:
                    if self.typed_graph_encoder is not None:
                        if not hasattr(batch_data, "edge_type"):
                            raise ValueError("typed_rgcn requires edge_type in the graph batch")
                        h_graph, graph_nodes = self.typed_graph_encoder(
                            graph_input_nodes,
                            graph_edge_index,
                            graph_edge_type,
                            batch_data.batch,
                            node_reliability=(node_quality if self.generated_reliability_mode != "none" else None),
                        )
                    else:
                        h_graph, graph_nodes = self.graph_model.encoder(
                            graph_input_nodes, batch_data.edge_index, batch_data.batch
                        )
                    if self.pooling_mode == "quality_gated_attention":
                        h_graph = quality_weighted_pool(
                            graph_nodes,
                            node_quality,
                            batch_data.batch,
                            int(text_embedding.size(0)),
                        )
                    edge_history = [(graph_edge_index, None)]
                h_graph = self.graph_model.graph_aligner(h_graph)
                if self.pooling_mode == "dual_modal_attention":
                    h_attention, node_text_attention, node_image_attention, _ = (
                        self.comment_attention(
                            graph_nodes,
                            batch_data.node_type,
                            batch_data.batch,
                            text_embedding,
                            image_embedding,
                        )
                    )
                    h_graph = self.graph_attention_norm(h_graph + h_attention)
                    node_source_attention = h_graph.new_zeros((batch_data.num_nodes,))
                    quality_alpha_real = h_graph.new_zeros(())
                    quality_alpha_generated = h_graph.new_zeros(())
                elif self.pooling_mode == "quality_gated_attention":
                    (
                        h_attention,
                        node_source_attention,
                        quality_alpha_real,
                        quality_alpha_generated,
                    ) = self.quality_attention(
                        graph_nodes,
                        batch_data.node_type,
                        node_quality,
                        batch_data.batch,
                        text_embedding,
                        image_embedding,
                    )
                    h_graph = self.graph_attention_norm(h_graph + h_attention)
                    node_text_attention = h_graph.new_zeros((batch_data.num_nodes,))
                    node_image_attention = h_graph.new_zeros((batch_data.num_nodes,))
                else:
                    node_text_attention = h_graph.new_zeros((batch_data.num_nodes,))
                    node_image_attention = h_graph.new_zeros((batch_data.num_nodes,))
                    node_source_attention = h_graph.new_zeros((batch_data.num_nodes,))
                    quality_alpha_real = h_graph.new_zeros(())
                    quality_alpha_generated = h_graph.new_zeros(())

        h_final = torch.cat((h_mm, h_graph), dim=-1)
        logits = self.classifier(h_final) if compute_logits else None
        global_logits = logits
        correction_output = {}
        if compute_logits and self.decision_mode == "global_local_correction":
            with torch.autocast(device_type=device_type, enabled=False):
                correction_output = self.local_correction(
                    h_final.float(), node_features.float(), batch_data.node_type,
                    batch_data.batch, root_feature.float(),
                    batch_data.root_attention_mask, image_feature_map.float(),
                )
                logits = global_logits.float() + correction_output["logit_delta"]
        return {
            "logits": logits,
            "global_logits": global_logits,
            "local_correction": correction_output,
            "h_mm": h_mm,
            "root_feature": root_feature,
            "image_feature_map": image_feature_map,
            "node_features": node_features,
            "text_embedding": text_embedding,
            "image_embedding": image_embedding,
            "h_graph": h_graph,
            "h_final": h_final,
            "node_text_attention": node_text_attention,
            "node_image_attention": node_image_attention,
            "node_quality": node_quality,
            "node_parent_semantic_gate": semantic_parent_gate,
            "generated_reliability_diagnostics": generated_reliability_diagnostics,
            "node_source_attention": node_source_attention,
            "quality_alpha_real": quality_alpha_real,
            "quality_alpha_generated": quality_alpha_generated,
            "edge_history": edge_history,
            "fusion_output": fusion_output,
            "evidence_output": evidence_output,
        }
