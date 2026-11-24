# Intent: evaluators and metrics

**Design decisions that apply** (`docs/wiki/design.md`). Test each one, and read its known gaps:
- D-10: splits by patient
- D-17: L2 norm
- D-18: test data stays out of training
- D-19: variance
- D-29: threshold
- D-32: bootstrap
- D-38: report sections
- D-40: `n_train`
- D-41: embedding cache
- D-46: `image_path` in a prediction
- D-47: fine-tune `loss="ce"`
- D-50: report scoring
- D-52: unused evaluator arguments
- Any other entry in the design.md section "Backbones and evaluators" that applies to this part.

**Other test steps:**

1. **Metrics** (`radharmony/evaluator/metrics/`): each metric gives the same value as sklearn, torchmetrics
   or the reference package on the same input. This applies to AUROC, AUPRC, F1, accuracy, Dice, IoU and
   the language metrics. Also test the edge cases: one class only in y_true, all-zero masks, empty
   predictions and ties. Macro and micro averages follow their documented definitions.
2. **Probes** (`classification/linear_probe.py`, `knn_probe.py`, `svm_probe.py`, `prototype_probe.py`):
   - The scaler and the hyperparameter search use train data only (D-18).
   - Check the default of `l2_normalize`, and what the benchmark scripts and notebooks pass (D-17).
   - `n_train` subsampling is stratified or random, as documented.
3. **Bootstrap** (`evaluator/sampling.py`, `classification/_probe_utils.py`): it uses `n_bootstrap` and
   `bootstrap_seed`. The CI or percentile is computed as documented. (What it draws: D-32. Point
   estimates: D-19.)
4. **Feature cache:** a cached feature file is used again only for the same backbone, dataset, split and
   transform. A changed setting must not load old features.
5. **Zero-shot** (`zero_shot.py`): the prompt for each label, the softmax or sigmoid choice and the label
   order all match the labels of the dataset.
6. **Fine-tune** (`finetune.py`): it trains only on the train split, selects the model on val, and reports
   on test (D-18).
7. **Segmentation** (`segmentation/*.py`): logits are upsampled to the mask size before Dice. Ignored
   classes are handled as documented. Background is included or excluded as documented.
8. **Report generation** (`language/report_generation.py`, `_report_parser.py`): the metrics compare a
   generated report with the report of the same study.

**How to test:** use small synthetic features and labels where you know the answer. Use a small fake
backbone (`torch.nn.Identity` on random tensors), and a few real `harmonized_df` rows when you need them.
Use the GPU from your task only when it is needed.
