<!-- Preserved from the project README before the collaboration rewrite on 2026-10-10. Historical instructions, not the current main experiment. -->

# MFCL 
  Source code for KBS accepted paper **M**ulti-modal Robustness **F**ake News Detection with **M**ross-Modal and Propagation Network Contrastive **L**earning
**C**ontrastive **L**earning(MFCL)
# Dataset
  The datasets used in the experiments were based on the two publicly available Weibo and PHEME datasets released by Zheng et al. (2023) and Zubiaga et al. (2017).
  Preprocessed datasets are available at: Weibo https://www.dropbox.com/scl/fi/o7mhk0chqeo21pugequh2/Chinese.rar?dl=0&rlkey=zba9vldtuu3np7olct2dbzzb7 and PHEME https://www.dropbox.com/scl/fi/3oh12ur58a8d62l5vm5cb/PHEME.rar?dl=0&rlkey=jwq68ru9l10nbphcg5z8pnf1z
# Dependencies
  Our code runs with the following packages installed:
  ```
    python 3.9.0
    torch 1.13.1
	cuda 11.6.1
	pytorch-cluster  1.6.0             
	pytorch-scatter  2.1.0          
	pytorch-sparse  0.6.15
	torchvision  0.14.1	
	numpy  1.23.5
	yaml  0.2.5
	pandas  1.5.2
    scikit-learn 1.0.2
 ```
 # Run
 Train and test
 ```
 python ./Weibo/test.py -c ./Weibo/configs/weibo.yaml
 ```

## Fakeddit with online encoders

The Fakeddit entry point consumes raw post titles, raw images and raw comments.
RoBERTa and ResNet-50 run inside the model during training; the dataset does not
need precomputed `root_feature`, `comment` or `img` arrays.

Download Fakeddit v2.0 from the
[official repository](https://github.com/entitize/Fakeddit) and arrange it as
follows:

```
data/fakeddit/
├── all_train.tsv
├── all_validate.tsv
├── all_test_public.tsv
├── comments.tsv
└── images/
    ├── <submission_id>.jpg
    └── ...
```

Use the official multimodal-only split when available. The expected TSV column
names and all paths can be changed in `Weibo/configs/fakeddit.yaml`. If images
are nested in subdirectories, enable `recursive_image_lookup` in that file.
Fakeddit's published binary convention is `0=fake, 1=true`; this pipeline keeps
that convention unchanged.

Install the additional dependencies and train:

```
pip install -r requirements-fakeddit.txt
python ./Weibo/train_fakeddit.py -c ./Weibo/configs/fakeddit.yaml
```

For an initial smoke run, set `data.max_samples: 1000` and
`pretrain_epochs: 1`. The online encoders use substantially more GPU memory than
the original precomputed features, so the default batch size is 4 with at most
8 comments per post.
