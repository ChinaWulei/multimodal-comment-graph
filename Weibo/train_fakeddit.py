"""Train MFCL on raw Fakeddit text, images and comment graphs.

Unlike the original entry point, this script performs feature extraction online,
pretrains only on the training split, selects checkpoints on validation data and
uses the test split exactly once at the end.
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from munch import DefaultMunch
from sklearn import metrics
from torch.cuda.amp import GradScaler, autocast
from torch_geometric.loader import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer

from fakeddit_dataset import (
    FakedditDataset,
    build_image_index,
    collect_split_ids,
    load_comment_map,
)
from model import MFCL
from online_encoder import OnlineFeatureEncoder
from propogation import ObjectiveLoss_Graph, ObjectiveLoss_TextImage
from propogation_function import get_augmented_by_drop_edge, get_augmented_by_node_mask


def parse_args():
    parser = argparse.ArgumentParser(description="Online MFCL training on Fakeddit")
    parser.add_argument("--config", "-c", required=True, help="YAML configuration file")
    parser.add_argument("--gpu", "-g", type=int, default=0, help="CUDA device index")
    parser.add_argument("--seed", "-s", type=int, default=100)
    parser.add_argument("--alpha", "-a", type=float, default=0.7)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_path(project_root, value):
    if value is None:
        return None
    path = Path(value)
    return str(path if path.is_absolute() else project_root / path)


def make_datasets(config, project_root):
    data_config = config.data
    train_tsv = resolve_path(project_root, data_config.train_tsv)
    val_tsv = resolve_path(project_root, data_config.val_tsv)
    test_tsv = resolve_path(project_root, data_config.test_tsv)
    images_dir = resolve_path(project_root, data_config.images_dir)
    comments_tsv = resolve_path(project_root, data_config.comments_tsv)

    tokenizer = AutoTokenizer.from_pretrained(config.encoder.text_model_name, use_fast=True)
    image_transform = transforms.Compose(
        [
            transforms.Resize(config.encoder.image_size + 32),
            transforms.CenterCrop(config.encoder.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )

    split_paths = (train_tsv, val_tsv, test_tsv)
    allowed_ids = collect_split_ids(split_paths, data_config.id_column)
    comment_map = load_comment_map(
        comments_tsv,
        allowed_submission_ids=allowed_ids,
        max_comments=data_config.max_comments,
    )
    image_index = (
        build_image_index(images_dir)
        if getattr(data_config, "recursive_image_lookup", False)
        else None
    )

    common = dict(
        images_dir=images_dir,
        tokenizer=tokenizer,
        image_transform=image_transform,
        comments_by_submission=comment_map,
        image_index=image_index,
        id_column=data_config.id_column,
        text_column=data_config.text_column,
        label_column=data_config.label_column,
        image_path_column=getattr(data_config, "image_path_column", None),
        max_text_length=data_config.max_text_length,
        max_comment_length=data_config.max_comment_length,
        max_comments=data_config.max_comments,
        min_comments=data_config.min_comments,
        undirected_graph=data_config.undirected_graph,
        max_samples=getattr(data_config, "max_samples", None),
    )
    return (
        FakedditDataset(tsv_path=train_tsv, **common),
        FakedditDataset(tsv_path=val_tsv, **common),
        FakedditDataset(tsv_path=test_tsv, **common),
    )


def make_loaders(config, datasets, device):
    if len(datasets[0]) < config.batch_size:
        raise ValueError(
            f"Training split has {len(datasets[0])} usable samples, fewer than "
            f"batch_size={config.batch_size}. Reduce batch_size or check image/comment paths."
        )
    common = dict(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    return (
        DataLoader(datasets[0], shuffle=True, drop_last=True, **common),
        DataLoader(datasets[1], shuffle=False, **common),
        DataLoader(datasets[2], shuffle=False, **common),
    )


def attach_online_features(batch_data, feature_encoder):
    root_feature, img, node_features = feature_encoder(batch_data)
    batch_data.root_feature = root_feature
    batch_data.img = img
    batch_data.x = node_features
    return batch_data


def parameter_groups(config, feature_encoder, *heads):
    head_parameters = []
    for module in heads:
        head_parameters.extend(parameter for parameter in module.parameters() if parameter.requires_grad)
    return [
        {
            "params": [parameter for parameter in feature_encoder.parameters() if parameter.requires_grad],
            "lr": config.encoder.learning_rate,
        },
        {"params": head_parameters, "lr": config.learning_rate},
    ]


def pretrain(config, loader, feature_encoder, graph_model, text_image_model, device):
    optimizer = torch.optim.AdamW(
        parameter_groups(config, feature_encoder, graph_model, text_image_model),
        weight_decay=config.weight_decay,
    )
    use_amp = bool(config.use_amp and device.type == "cuda")
    scaler = GradScaler(enabled=use_amp)

    for epoch in range(config.pretrain_epochs):
        feature_encoder.train()
        graph_model.train()
        text_image_model.train()
        total_loss = 0.0
        seen = 0
        for batch_data in loader:
            batch_data = batch_data.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                batch_data = attach_online_features(batch_data, feature_encoder)
                augmented_nodes = get_augmented_by_node_mask(batch_data)
                augmented_edges = get_augmented_by_drop_edge(batch_data)
                z1, z2 = graph_model(augmented_nodes, augmented_edges)
                graph_loss = graph_model.cal_loss(z1, z2)

                text, image = text_image_model(batch_data)
                multimodal_loss = text_image_model.cal_loss_t(text, image)
                loss = graph_loss + config.multimodal_loss_weight * multimodal_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_size = batch_data.y.numel()
            total_loss += loss.detach().item() * batch_size
            seen += batch_size
        print(f"pretrain epoch={epoch + 1} loss={total_loss / max(seen, 1):.6f}")


def classification_step(batch_data, feature_encoder, graph_model, text_image_model, classifier):
    batch_data = attach_online_features(batch_data, feature_encoder)
    graph_embedding, _ = graph_model.encoder(
        batch_data.x, batch_data.edge_index, batch_data.batch
    )
    graph_embedding = graph_model.graph_aligner(graph_embedding)
    text_embedding, image_embedding = text_image_model(batch_data)
    logits = classifier(text_embedding, image_embedding, graph_embedding)
    return logits


def evaluate(loader, feature_encoder, graph_model, text_image_model, classifier, device, use_amp):
    feature_encoder.eval()
    graph_model.eval()
    text_image_model.eval()
    classifier.eval()
    predictions, labels = [], []
    total_loss = 0.0
    seen = 0
    with torch.no_grad():
        for batch_data in loader:
            batch_data = batch_data.to(device)
            with autocast(enabled=use_amp):
                logits = classification_step(
                    batch_data, feature_encoder, graph_model, text_image_model, classifier
                )
                y = batch_data.y.view(-1)
                loss = F.cross_entropy(logits, y)
            prediction = logits.argmax(dim=1)
            predictions.append(prediction.cpu().numpy())
            labels.append(y.cpu().numpy())
            total_loss += loss.item() * y.numel()
            seen += y.numel()

    prediction = np.concatenate(predictions)
    label = np.concatenate(labels)
    return {
        "loss": total_loss / max(seen, 1),
        "accuracy": metrics.accuracy_score(label, prediction),
        "precision_weighted": metrics.precision_score(label, prediction, average="weighted", zero_division=0),
        "recall_weighted": metrics.recall_score(label, prediction, average="weighted", zero_division=0),
        "f1_weighted": metrics.f1_score(label, prediction, average="weighted", zero_division=0),
        "f1_macro": metrics.f1_score(label, prediction, average="macro", zero_division=0),
        "report": metrics.classification_report(label, prediction, digits=4, zero_division=0),
    }


def save_checkpoint(path, epoch, feature_encoder, graph_model, text_image_model, classifier, metrics_value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "feature_encoder": feature_encoder.state_dict(),
            "graph_model": graph_model.state_dict(),
            "text_image_model": text_image_model.state_dict(),
            "classifier": classifier.state_dict(),
            "validation_metrics": metrics_value,
        },
        path,
    )


def load_checkpoint(path, feature_encoder, graph_model, text_image_model, classifier, device):
    checkpoint = torch.load(path, map_location=device)
    feature_encoder.load_state_dict(checkpoint["feature_encoder"])
    graph_model.load_state_dict(checkpoint["graph_model"])
    text_image_model.load_state_dict(checkpoint["text_image_model"])
    classifier.load_state_dict(checkpoint["classifier"])


def train_classifier(
    config,
    train_loader,
    val_loader,
    feature_encoder,
    graph_model,
    text_image_model,
    classifier,
    device,
    checkpoint_path,
):
    optimizer = torch.optim.AdamW(
        parameter_groups(config, feature_encoder, graph_model, text_image_model, classifier),
        weight_decay=config.weight_decay,
    )
    use_amp = bool(config.use_amp and device.type == "cuda")
    scaler = GradScaler(enabled=use_amp)
    best_f1 = -1.0
    stale_epochs = 0

    for epoch in range(config.max_epoch):
        feature_encoder.train()
        graph_model.train()
        text_image_model.train()
        classifier.train()
        total_loss = 0.0
        seen = 0
        for batch_data in train_loader:
            batch_data = batch_data.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                logits = classification_step(
                    batch_data, feature_encoder, graph_model, text_image_model, classifier
                )
                y = batch_data.y.view(-1)
                loss = F.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.detach().item() * y.numel()
            seen += y.numel()

        val_metrics = evaluate(
            val_loader,
            feature_encoder,
            graph_model,
            text_image_model,
            classifier,
            device,
            use_amp,
        )
        print(
            f"train epoch={epoch + 1} loss={total_loss / max(seen, 1):.6f} "
            f"val_acc={val_metrics['accuracy']:.4f} val_macro_f1={val_metrics['f1_macro']:.4f}"
        )
        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            stale_epochs = 0
            save_checkpoint(
                checkpoint_path,
                epoch,
                feature_encoder,
                graph_model,
                text_image_model,
                classifier,
                val_metrics,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                print(f"early stopping after {epoch + 1} epochs")
                break


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    seed_everything(args.seed)
    with open(args.config, "r", encoding="utf-8") as stream:
        config = DefaultMunch.fromDict(yaml.safe_load(stream))

    project_root = Path(__file__).resolve().parents[1]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint_path = resolve_path(project_root, config.checkpoint_path)

    datasets = make_datasets(config, project_root)
    loaders = make_loaders(config, datasets, device)
    print(
        f"device={device} train={len(datasets[0])} "
        f"validation={len(datasets[1])} test={len(datasets[2])}"
    )

    feature_encoder = OnlineFeatureEncoder(
        text_model_name=config.encoder.text_model_name,
        output_dim=config.contextual_transform.output_dim,
        pretrained_image=config.encoder.pretrained_image,
        fine_tune_text=config.encoder.fine_tune_text,
        fine_tune_image=config.encoder.fine_tune_image,
        node_encode_batch_size=config.encoder.node_encode_batch_size,
    ).to(device)
    graph_model = ObjectiveLoss_Graph(num_hidden=256, num_gc_layers=3).to(device)
    text_image_model = ObjectiveLoss_TextImage(config).to(device)
    classifier = MFCL(config, args.alpha).to(device)

    if config.pretrain_epochs > 0:
        pretrain(config, loaders[0], feature_encoder, graph_model, text_image_model, device)
    train_classifier(
        config,
        loaders[0],
        loaders[1],
        feature_encoder,
        graph_model,
        text_image_model,
        classifier,
        device,
        checkpoint_path,
    )
    load_checkpoint(
        checkpoint_path,
        feature_encoder,
        graph_model,
        text_image_model,
        classifier,
        device,
    )
    test_metrics = evaluate(
        loaders[2],
        feature_encoder,
        graph_model,
        text_image_model,
        classifier,
        device,
        bool(config.use_amp and device.type == "cuda"),
    )
    print("\nTest classification report")
    print(test_metrics.pop("report"))
    print("Test metrics:", test_metrics)


if __name__ == "__main__":
    main()
