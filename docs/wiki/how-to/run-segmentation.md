# Run segmentation

A segmentation probe trains a small head on the frozen patch features of an encoder, and scores the predicted
masks. This page segments the lungs of [Montgomery County
CXR](../reference/datasets/cxr-boxes-masks/montgomery_cxr.md) (138 images) with RAD-DINO. It runs the linear
head first, then the conv and the UPerNet heads. The 3 heads take 2-D images and masks only. So a dataset with
3-D masks, for example RSNA 2022 Cervical Spine, does not fit them.

Before you start, install the `raddino` extra, and
[get the Montgomery data](get-the-data.md#montgomery-county-cxr).

## 1. Build the encoder and the dataset

Pass `output_keys={"img", "mask"}` to the backbone recipe. Then the encoder gives patch features,
`Tensor[B, D, H', W']`, and the transform resizes the mask with the image
([Segmentation mode](../reference/backbones/index.md#segmentation-mode)). Montgomery has a left and a right
lung mask for each image. The dataset joins them into one mask file in `mask_output_dir`.

<!-- block: run=gpu timeout=600 -->
```python
from radharmony.dataset import MontgomeryCXRDataset
from radharmony.evaluator.backbones import make_raddino

transform, encoder = make_raddino(device="cuda", output_keys={"img", "mask"})
ds = MontgomeryCXRDataset(
    base_image_dir="/data/MontgomerySet/CXR_png/",
    transform=transform,
    output_mask=True,
    mask_output_dir="/data/montgomery_lung_masks/",  # any folder you can write to
    cache_dir=None,
)
sample = ds.get_datasets()[0]
print(sample["img"].shape, sample["mask"].shape)  # torch.Size([3, 518, 518]) torch.Size([1, 518, 518])
```

## 2. Run the linear head

`num_classes=2` is the background and the lung. A short run uses 2 folds and 3 epochs. `num_workers=8`
reads the large PNG files faster.

<!-- block: run=gpu timeout=900 -->
```python
from radharmony.evaluator import LinearProbeSegEvaluator

ev = LinearProbeSegEvaluator(
    encoder,
    dataset=ds,
    num_classes=2,
    n_folds=2,
    epochs=3,
    num_workers=8,
    output_dir="outputs/linear_probe_seg",
)
df = ev.evaluate()
ev.save_results(df)  # writes results.csv and results_summary.csv
print(df[df["label"] == "macro_average"][["fold", "dice", "iou"]].round(3))
#    fold   dice    iou
# 4     0  0.868  0.769
# 5     1  0.854  0.749
```

In a real run on a shared GPU, this step took about 4 minutes. With `cache_dir=None`, the dataset reads
and resizes each image again in each epoch.

## 3. Read the result table

The table has one row for each class and fold. `class_0` is the background and `class_1` is the lung.
`macro_average` is the mean over the classes other than `class_0`, so here it is the lung score. Each
metric is a mean over the test images ([Metrics](../reference/evaluators/linear_probe_seg.md#metrics)).

<!-- block: run=gpu timeout=600 -->
```python
print(df["label"].unique().tolist())  # ['class_0', 'class_1', 'macro_average']
print(round(df.loc[df["label"] == "class_1", "dice"].mean(), 3))  # 0.861, the mean lung Dice of the 2 folds
```

## 4. Switch to a larger head

The conv and the UPerNet heads take the same arguments, and some of their own. `conv_hidden_size` sets
the channels of the conv head. `upernet_hidden_size` and `upernet_pool_scales` set the UPerNet head.

<!-- block: run=gpu timeout=1500 -->
```python
from radharmony.evaluator import ConvProbeSegEvaluator, UPerNetSegEvaluator

heads = {
    "conv": (ConvProbeSegEvaluator, {"conv_hidden_size": 256}),
    "upernet": (UPerNetSegEvaluator, {"upernet_hidden_size": 256, "upernet_pool_scales": (1, 2, 3, 6)}),
}
for name, (probe, head_args) in heads.items():
    ev = probe(
        encoder,
        dataset=ds,
        num_classes=2,
        n_folds=2,
        epochs=3,
        num_workers=8,
        output_dir=f"outputs/{name}_seg",
        **head_args,
    )
    df = ev.evaluate()
    ev.save_results(df)
    print(name, round(df.loc[df["label"] == "macro_average", "dice"].mean(), 3))
# conv 0.95
# upernet 0.959
```

In the real run, the 2 heads took about 7 minutes. With 3 epochs, the larger heads give a higher Dice
than the linear head.

The UPerNet head needs a feature size `D` that is a multiple of 32
([UPerNet](../reference/evaluators/upernet_seg.md)).

## Check

- `outputs/linear_probe_seg/`, `outputs/conv_seg/` and `outputs/upernet_seg/` each have `results.csv`
  and `results_summary.csv`.
- The lung Dice is above 0.8 for each head. The lungs are large and easy to find. A small target, for
  example a pneumothorax, gives a much lower Dice.

## See also

- [Linear probe (segmentation)](../reference/evaluators/linear_probe_seg.md): early stopping, the metrics,
  and how to save the head and predict masks.
- [Conv probe](../reference/evaluators/conv_probe_seg.md) and [UPerNet](../reference/evaluators/upernet_seg.md):
  the head layers and their arguments.
- [Run a probe](run-a-probe.md): the classification probes.
