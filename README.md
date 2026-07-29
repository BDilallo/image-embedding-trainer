# image-embedding-trainer

Trains an image embedding model (any [timm](https://github.com/huggingface/pytorch-image-models)
backbone) using a metric-learning loss, for retrieval/verification-style tasks
(e.g. "is this the same X as that other image?"). Not tied to any particular
subject — faces, characters, products, animals, documents, etc. all work as
long as your data follows the folder convention below.

## Data format

Both `--train_dir` and `--val_dir` must be organized as one subfolder per
class, with images directly inside each subfolder:

```
train_dir/
    class_1/
        image1.png
        image2.png
    class_2/
        image1.png
        ...

val_dir/
    class_1/
        image1.png
    class_2/
        image1.png
    ...
```

The subfolder name is the class label; classes in `val_dir` do **not** need
to overlap with classes in `train_dir` — validation measures open-set
retrieval quality (can the model group same-class images it never saw during
training?), which is the standard way to evaluate an embedding model.

Each class needs at least 2 images in both `train_dir` and `val_dir` for the
loss/metrics to be meaningful (pairs are needed to define "same class").

## Install

```
pip install -e .
```

## Train

```
python -m image_embedding_trainer.train \
    --train_dir path/to/train \
    --val_dir path/to/val \
    --out_dir path/to/output
```

Checkpoints (`best_model.pt`, `last_model.pt`), `config.json`, and
`metrics_history.json` are written to `--out_dir`. Resume a run with:

```
python -m image_embedding_trainer.train ... --resume path/to/output/last_model.pt
```

## Adapting to your data

The defaults (RGB input, ImageNet normalization, horizontal flip + mild
color jitter, ArcFace loss) suit natural RGB photos with left-right symmetric
subjects — think faces or characters. Everything below is overridable via
CLI flags so you shouldn't need to edit source for a different kind of data.

**Loss function** (`--loss_name`): `arcface` (default), `subcenter_arcface`,
`triplet`, or `contrastive`. ArcFace-family losses assume a fixed, closed set
of classes at train time with several images per class; triplet/contrastive
don't need per-class weights and can suit smaller or more irregular datasets.
Loss-specific hyperparameters: `--arcface_margin`, `--arcface_scale`,
`--arcface_sub_centers`, `--triplet_margin`, `--contrastive_pos_margin`,
`--contrastive_neg_margin`.

**Image channels** (`--num_channels`): `3` (RGB, default) or `1` (grayscale).
When set to `1`, also pass matching `--normalize_mean`/`--normalize_std`
(single value each), since the ImageNet defaults are 3-channel.

**Augmentation**: `--no_horizontal_flip` (disable flip for domains where
mirroring changes identity, e.g. text or asymmetric marks),
`--no_color_jitter` (disable for domains where color is discriminative,
e.g. distinguishing color variants of the same product), or tune
`--color_jitter_brightness/_contrast/_saturation` and
`--random_crop_scale_min/_max` directly.

**Backbone** (`--model_name`): any model name recognized by `timm.create_model`.

Run `python -m image_embedding_trainer.train --help` for the full list of
options.
