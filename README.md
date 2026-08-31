# Plant Disease Classification with Dual-Backbone Representation-Aware Ensemble

## Research-Oriented Kaggle Pipeline for PlantVillage `.npy` Data on Dual NVIDIA T4 GPUs

This repository implements a reproducible research pipeline for multi-class plant disease classification using the PlantVillage dataset stored as `uint8` NumPy arrays. The framework is designed to investigate whether **backbone diversity** and **input-representation diversity** can improve classification performance beyond a single CNN model.

The study evaluates two complementary convolutional backbones:

- **ConvNeXt-Tiny**
- **EfficientNet-B0**

under two image representations:

- **Color**
- **Segmented**

The final system selects the best representation for each backbone using the validation set and combines the selected models through a **validation-optimized weighted probability ensemble**.

---

## 1. Research Motivation

Plant disease recognition is often evaluated on controlled datasets such as PlantVillage, where modern convolutional networks can achieve very high classification accuracy. However, high accuracy alone does not explain:

- whether a model benefits more from raw color information or background-removed segmented leaves,
- whether modern CNN backbones make complementary errors,
- whether attention and adaptive pooling improve disease-related feature extraction,
- whether class imbalance affects minority disease classes,
- and whether model fusion provides a consistent gain over the strongest individual network.

This project is therefore designed as a **comparative and ablation-oriented study**, rather than only as a single high-accuracy classifier.

The main research hypothesis is:

> A representation-aware ensemble that combines complementary convolutional backbones can improve robustness and macro-level classification performance when the component models learn partially different disease cues.

---

## 2. Research Questions

The pipeline is structured around the following research questions.

### RQ1 — Backbone comparison

How does ConvNeXt-Tiny compare with EfficientNet-B0 for PlantVillage disease classification under an identical split and optimization protocol?

### RQ2 — Representation comparison

Does the original color representation or the segmented representation provide stronger discriminative information for each backbone?

### RQ3 — Custom feature aggregation

Can lightweight attention and learnable generalized mean pooling improve feature discrimination relative to the default classification head?

### RQ4 — Class imbalance

Does class-balanced focal supervision improve macro-level performance across disease classes with different sample frequencies?

### RQ5 — Ensemble complementarity

Do the best ConvNeXt and EfficientNet models make sufficiently different errors to justify probability-level ensemble fusion?

### RQ6 — Weighted fusion

Does a validation-selected ensemble weight outperform simple equal averaging and the strongest individual model?

---

## 3. Dataset Organization

The expected Kaggle dataset root is:

```text
/kaggle/input/datasets/leminhhung0101/plantvillage-npy-dataset
```

Expected structure:

```text
plantvillage-npy-dataset/
├── color/
│   ├── Apple___Apple_scab.npy
│   ├── Apple___Black_rot.npy
│   ├── ...
│   └── Tomato___healthy.npy
│
├── segmented/
│   ├── Apple___Apple_scab.npy
│   ├── Apple___Black_rot.npy
│   ├── ...
│   └── Tomato___healthy.npy
│
└── grayscale/
    └── ...
```

Each class file should contain multiple images, typically:

```python
shape = (N, 224, 224, 3)
dtype = uint8
```

The current research pipeline uses:

```text
color
segmented
```

while grayscale can be retained for future ablation studies.

---

## 4. Robust Representation Matching

The training script does not assume that every `.npy` file is perfectly matched.

Class files are matched using their filename stem.

Example:

```text
color/
├── A.npy
├── B.npy
└── C.npy

segmented/
├── A.npy
├── B.npy
└── D.npy
```

Only:

```text
A.npy
B.npy
```

are used.

Unmatched files are ignored and recorded in:

```text
/kaggle/working/plantvillage_t4x2/splits/unmatched_files.json
```

If two matching class arrays have different lengths:

```text
color/A.npy       = 1500 samples
segmented/A.npy   = 1492 samples
```

the pipeline uses:

```text
min(1500, 1492) = 1492 paired indices
```

and records the difference in:

```text
pairing_report.csv
```

The source files under `/kaggle/input` are never modified.

### Important pairing assumption

For every matched class:

```text
color/class.npy[i]
```

and:

```text
segmented/class.npy[i]
```

must correspond to the same original source image.

This assumption should be guaranteed during the `.npy` conversion stage.

---

## 5. Data Split Protocol

The dataset is divided using a fixed stratified:

```text
Train      80%
Validation 10%
Test       10%
```

with:

```text
seed = 42
```

Splitting is performed independently inside every disease class.

The same sample index is assigned to the same subset for both color and segmented representations.

Example output from the current dataset:

```text
Train      43,429
Validation  5,417
Test        5,459
Classes        38
```

Split manifests are stored as:

```text
/kaggle/working/plantvillage_t4x2/splits/
├── train/
│   └── manifest.csv
├── valid/
│   └── manifest.csv
├── test/
│   └── manifest.csv
├── class_to_idx.json
├── split_meta.json
├── unmatched_files.json
└── pairing_report.csv
```

The pipeline stores indices rather than duplicating all NumPy image arrays into separate folders. This significantly reduces Kaggle storage usage while preserving a reproducible split.

---

## 6. Experimental Design

Four primary experiments are trained:

| ID | Backbone | Representation |
|---|---|---|
| E1 | ConvNeXt-Tiny | Color |
| E2 | ConvNeXt-Tiny | Segmented |
| E3 | EfficientNet-B0 | Color |
| E4 | EfficientNet-B0 | Segmented |

This design avoids pre-assuming that one representation is always superior for a given backbone.

The validation set determines:

1. the best representation for ConvNeXt-Tiny,
2. the best representation for EfficientNet-B0.

Only after these decisions are locked is the final ensemble constructed.

---

## 7. ConvNeXt-Tiny Branch

The ConvNeXt branch is:

```text
Input image
    ↓
ConvNeXt-Tiny backbone
    ↓
Final convolutional feature map
    ↓
Efficient Channel Attention (ECA)
    ↓
Generalized Mean Pooling (GeM)
    ↓
Linear
    ↓
Batch Normalization
    ↓
ReLU
    ↓
Dropout
    ↓
Final classifier
```

### 7.1 Efficient Channel Attention

ECA is used as a lightweight channel recalibration mechanism.

Conceptually:

\[
X' = X \odot A_c(X)
\]

where:

- \(X\) is the feature map,
- \(A_c\) is the learned channel attention,
- \(\odot\) denotes channel-wise multiplication.

ECA is intended to emphasize feature channels associated with discriminative visual patterns such as:

- lesion color,
- necrotic areas,
- disease texture,
- discoloration,
- mildew-like structures,
- and boundary changes.

---

## 8. EfficientNet-B0 Branch

The EfficientNet branch is:

```text
Input image
    ↓
EfficientNet-B0 backbone
    ↓
Final convolutional feature map
    ↓
CBAM
    ↓
Generalized Mean Pooling
    ↓
Linear
    ↓
Batch Normalization
    ↓
ReLU
    ↓
Dropout
    ↓
Final classifier
```

### 8.1 CBAM

The Convolutional Block Attention Module applies:

```text
Channel attention
        ↓
Spatial attention
```

The resulting representation is:

\[
X' = A_s(A_c(X)\odot X)
\]

where the network learns both:

- **what feature channels are important**, and
- **where important regions occur spatially**.

The late placement of CBAM is intentional. It provides additional attention capacity without inserting attention modules into every EfficientNet stage.

---

## 9. Generalized Mean Pooling

Both branches use learnable **GeM pooling** instead of a fixed Global Average Pooling layer.

For feature map activations \(x_i\):

\[
\mathrm{GeM}(x)
=
\left(
\frac{1}{N}
\sum_{i=1}^{N}
x_i^p
\right)^{1/p}
\]

where \(p\) is learned during training.

Initialization:

```text
p = 3.0
```

with the implementation constraining it to a numerically stable interval.

Compared with standard GAP:

```text
p = 1
```

larger learned values allow the network to place more emphasis on highly activated local disease regions.

The learned value of `GeM_p` is recorded in the training history.

---

## 10. Classification Head

The custom classification head is:

```text
Feature vector
    ↓
Linear
    ↓
BatchNorm
    ↓
ReLU
    ↓
Dropout = 0.30
    ↓
Linear(num_classes)
```

This head is deliberately lightweight so that most representation capacity remains in the pretrained backbone.

---

## 11. Class-Balanced Focal Loss

PlantVillage contains unequal numbers of samples across disease categories.

Instead of using raw inverse-frequency weighting, the pipeline uses the **effective number of samples** formulation.

For class \(c\):

\[
w_c =
\frac{1-\beta}
{1-\beta^{n_c}}
\]

where:

- \(n_c\) is the number of training samples in class \(c\),
- \(\beta\) controls the strength of class reweighting.

Default:

```text
beta = 0.999
```

Weights are normalized so that:

\[
\mathrm{mean}(w_c)=1
\]

The class-balanced weight is combined with focal modulation:

\[
L =
-w_y
(1-p_y)^\gamma
\log(p_y)
\]

with:

```text
gamma = 2.0
```

This encourages the model to focus more strongly on:

- minority classes,
- difficult disease patterns,
- and incorrectly classified samples,

while reducing the influence of very easy examples.

Importantly, class weights are computed using **training data only**.

---

## 12. Progressive Fine-Tuning

Training is divided into two stages.

### Stage 1 — Head warm-up

Default:

```text
3 epochs
```

The pretrained backbone is frozen:

```text
Backbone      frozen
Attention     trainable
GeM           trainable
Classifier    trainable
```

Learning rate:

```text
3e-4
```

This allows newly initialized custom layers to adapt before updating pretrained ImageNet features.

### Stage 2 — Full fine-tuning

Default maximum:

```text
15 epochs
```

The backbone is unfrozen.

Differential learning rates are used:

```text
Pretrained backbone  3e-5
Custom layers/head   3e-4
```

Thus:

\[
LR_{head} = 10 \times LR_{backbone}
\]

This reduces the risk of rapidly destroying useful pretrained representations.

---

## 13. Optimization

Default training configuration:

```text
Optimizer            AdamW
Weight decay         1e-4
Batch size           32 / GPU
Image size           224 × 224
AMP                   enabled
Gradient clipping    5.0
Scheduler            CosineAnnealingLR
Early stopping       patience = 5
Seed                  42
```

The best checkpoint is selected using:

```text
Primary metric:
Validation Macro-F1

Tie-break:
Validation Accuracy
```

The test set is not used for checkpoint selection.

---

## 14. Data Augmentation

Training augmentation includes:

```text
RandomResizedCrop
Horizontal flip
Vertical flip
Rotation
Color jitter
ImageNet normalization
```

Color images receive slightly stronger photometric augmentation than segmented images.

Validation and test preprocessing contain no random augmentation.

---

## 15. Dual NVIDIA T4 Training

The script is designed for:

```text
NVIDIA T4 ×2
```

The four experiments are trained in two groups.

### Group 1

```text
GPU 0 → ConvNeXt-Tiny + Color
GPU 1 → ConvNeXt-Tiny + Segmented
```

### Group 2

```text
GPU 0 → EfficientNet-B0 + Color
GPU 1 → EfficientNet-B0 + Segmented
```

This strategy runs independent experiments concurrently instead of applying multi-GPU data parallelism to a single model.

It is especially suitable for an ablation study where all models must be trained independently.

---

## 16. Validation-Based Model Selection

After all four models are trained:

```text
ConvNeXt + Color
ConvNeXt + Segmented
EffNet   + Color
EffNet   + Segmented
```

the system selects:

```text
Best ConvNeXt representation
Best EfficientNet representation
```

using validation Macro-F1.

This prevents representation choice from being decided by the test set.

---

## 17. Weighted Probability Ensemble

The selected models output class probability vectors:

\[
P_C \in \mathbb{R}^{K}
\]

and:

\[
P_E \in \mathbb{R}^{K}
\]

where \(K=38\).

The final prediction is:

\[
P_{final}
=
\alpha P_C
+
(1-\alpha)P_E
\]

The ensemble coefficient is searched on the validation set:

```text
alpha = 0.00
alpha = 0.02
alpha = 0.04
...
alpha = 0.98
alpha = 1.00
```

The selected weight maximizes:

```text
Validation Macro-F1
```

with validation accuracy used as a tie-break.

After \(\alpha\) is fixed, the test set is evaluated once.

This avoids test-set tuning.

---

## 18. Complementarity Analysis

An ensemble should not be retained solely because it contains two strong models.

The pipeline additionally measures:

```text
both_correct
convnext_only_correct
effnet_only_correct
both_wrong
disagreement_rate
```

This helps determine whether the two models actually provide complementary decision boundaries.

For example:

```text
ConvNeXt wrong / EfficientNet correct
```

and:

```text
EfficientNet wrong / ConvNeXt correct
```

represent the samples where ensemble fusion may provide genuine benefit.

---

## 19. Evaluation Metrics

The study reports:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1
- Weighted F1
- Per-class Precision
- Per-class Recall
- Per-class F1
- Confusion Matrix

For the final ensemble, bootstrap confidence intervals can also be reported for:

- Accuracy
- Macro F1

Macro-F1 is treated as a primary metric because it gives equal importance to all disease classes.

---

## 20. Checkpoint and Output Structure

Training outputs are written to:

```text
/kaggle/working/plantvillage_t4x2/
```

Expected structure:

```text
plantvillage_t4x2/
│
├── splits/
│   ├── train/
│   │   └── manifest.csv
│   ├── valid/
│   │   └── manifest.csv
│   ├── test/
│   │   └── manifest.csv
│   ├── class_to_idx.json
│   ├── split_meta.json
│   ├── unmatched_files.json
│   └── pairing_report.csv
│
├── runs/
│   ├── convnext_color/
│   │   ├── best.pt
│   │   ├── last.pt
│   │   ├── history.csv
│   │   └── console.log
│   │
│   ├── convnext_segmented/
│   ├── effnet_color/
│   └── effnet_segmented/
│
├── ensemble/
│   ├── all_experiments_valid.csv
│   ├── selected_models.json
│   ├── alpha_search_valid.csv
│   ├── ensemble_config.json
│   ├── final_results.json
│   ├── classification_report.csv
│   ├── test_predictions.csv
│   └── confusion_matrix.png
│
└── test_predictions/
    ├── convnext_color/
    ├── convnext_segmented/
    ├── effnet_color/
    ├── effnet_segmented/
    ├── ensemble/
    └── all_models_test_summary.csv
```

---

## 21. Training Script

Main training script:

```text
train_plantvillage_t4x2_ROBUST_MATCH.py
```

### Prepare the split

```bash
python train_plantvillage_t4x2_ROBUST_MATCH.py --mode prepare
```

### Train all four experiments on T4×2

```bash
python train_plantvillage_t4x2_ROBUST_MATCH.py --mode train-all
```

### View validation summary

```bash
python train_plantvillage_t4x2_ROBUST_MATCH.py --mode summary
```

### Select the best representation for each backbone

```bash
python train_plantvillage_t4x2_ROBUST_MATCH.py --mode select
```

### Construct the final ensemble

```bash
python train_plantvillage_t4x2_ROBUST_MATCH.py --mode ensemble
```

### Run the complete pipeline

```bash
python train_plantvillage_t4x2_ROBUST_MATCH.py --mode all
```

---

## 22. Testing Saved Models

Standalone test script:

```text
test_plantvillage_models.py
```

### Evaluate all available individual models

```bash
python test_plantvillage_models.py --mode all-models
```

### Evaluate a specific model

Example:

```bash
python test_plantvillage_models.py \
    --mode predict-one \
    --backbone convnext \
    --representation color
```

### Evaluate the saved weighted ensemble

```bash
python test_plantvillage_models.py --mode ensemble
```

The ensemble test script loads:

```text
ensemble_config.json
```

and therefore reuses the previously locked model selection and fusion weights rather than optimizing them again.

---

## 23. Recommended Ablation Study

For a research paper, the final ensemble should not be reported without controlled ablations.

A minimum ablation table should contain:

| ID | Model | Representation | Attention | Pooling | Loss | Fusion |
|---|---|---|---|---|---|---|
| A1 | ConvNeXt-Tiny | Color | None | GAP | CE | — |
| A2 | ConvNeXt-Tiny | Color | ECA | GeM | CB-Focal | — |
| A3 | ConvNeXt-Tiny | Segmented | ECA | GeM | CB-Focal | — |
| B1 | EfficientNet-B0 | Color | None | GAP | CE | — |
| B2 | EfficientNet-B0 | Color | CBAM | GeM | CB-Focal | — |
| B3 | EfficientNet-B0 | Segmented | CBAM | GeM | CB-Focal | — |
| E1 | Best ConvNeXt + Best EffNet | selected | custom | GeM | — | 0.5 / 0.5 |
| E2 | Best ConvNeXt + Best EffNet | selected | custom | GeM | — | validation weighted |

This allows the study to distinguish gains caused by:

- backbone architecture,
- representation,
- attention,
- pooling,
- loss reweighting,
- and ensemble fusion.

---

## 24. Recommended Statistical Reporting

For the final paper, report at least:

```text
Mean performance
95% confidence interval
Per-class F1
Confusion matrix
Class distribution
Model parameter count
Inference cost
```

If computational budget permits, repeat training with multiple seeds such as:

```text
42
123
3407
```

and report:

\[
\text{mean} \pm \text{standard deviation}
\]

This is preferable to drawing strong conclusions from a single initialization.

---

## 25. Important Limitation of PlantVillage

PlantVillage contains relatively controlled image conditions.

Consequently, extremely high test accuracy on an internal PlantVillage split does not necessarily imply reliable performance in real-world agricultural environments.

Potential deployment domain shifts include:

- uncontrolled illumination,
- complex backgrounds,
- partial leaves,
- overlapping leaves,
- motion blur,
- weather effects,
- different cameras,
- multiple simultaneous diseases,
- and disease severity variation.

Therefore, the strongest follow-up experiment is **external validation** on a field-image dataset such as PlantDoc or another independently collected dataset.

A research conclusion should distinguish:

```text
in-domain PlantVillage performance
```

from:

```text
cross-domain field robustness
```

---

## 26. Suggested Paper Contribution Framing

A defensible contribution statement could be:

> We investigate a representation-aware dual-backbone plant disease classification framework that jointly studies raw color and segmented leaf representations under a shared experimental protocol. The framework combines ConvNeXt-Tiny and EfficientNet-B0 with lightweight attention, learnable generalized mean pooling, class-balanced focal supervision, and validation-optimized late probability fusion. Rather than assigning a fixed representation to each backbone, the proposed protocol first evaluates all backbone-representation combinations, selects the strongest complementary branches using validation data, and quantifies prediction diversity before final ensemble evaluation.

Avoid claiming that the framework is universally superior unless supported by:

- controlled ablations,
- repeated runs,
- statistical testing,
- and preferably external validation.

---

## 27. Key Research Safeguards

The implementation follows the following safeguards:

1. **Subject/image split decisions are made before training.**
2. **Class weights are computed from training data only.**
3. **Checkpoint selection uses validation data only.**
4. **Representation selection uses validation data only.**
5. **Ensemble weight selection uses validation data only.**
6. **The test set is not used to tune hyperparameters.**
7. **Source `.npy` files under `/kaggle/input` are never modified.**
8. **Unmatched representation files are logged and ignored.**
9. **All experiment checkpoints are stored independently.**
10. **Individual-model results are retained even when ensemble fusion is used.**

---

## 28. Reproducibility Checklist

Before reporting results, record:

```text
Dataset version
Dataset path
Number of classes
Number of train/valid/test samples
Random seed
PyTorch version
timm version
CUDA version
GPU type
Batch size
Learning rates
Weight decay
Number of epochs
Early-stopping patience
Class-balanced beta
Focal gamma
Selected checkpoints
Selected representations
Selected ensemble alpha
```

Kaggle hardware for the intended experiment:

```text
NVIDIA Tesla T4 ×2
```

---

## 29. Reference Methods

The architecture is conceptually informed by the following established methods:

1. Liu et al.  
   **A ConvNet for the 2020s**  
   ConvNeXt.

2. Tan and Le.  
   **EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks**  
   EfficientNet.

3. Wang et al.  
   **ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks**  
   Efficient Channel Attention.

4. Woo et al.  
   **CBAM: Convolutional Block Attention Module**  
   Channel and spatial attention.

5. Cui et al.  
   **Class-Balanced Loss Based on Effective Number of Samples**  
   Effective-number class reweighting.

6. Lin et al.  
   **Focal Loss for Dense Object Detection**  
   Focal modulation for hard-example emphasis.

7. Generalized Mean Pooling literature  
   Learnable pooling between average-like and max-like aggregation behavior.

The exact combination implemented in this project should be treated as an experimental framework whose contribution must be demonstrated empirically through ablation.

---

## 30. Current Research Pipeline

```text
PlantVillage uint8 .npy
        │
        ├────────────── Color ────────────────┐
        │                                     │
        │                              ConvNeXt-Tiny
        │                                     │
        │                                    ECA
        │                                     │
        │                                    GeM
        │                                     │
        │                                Classifier
        │                                     │
        │                                  P_conv
        │                                     │
        │                                     │
        └────────── Segmented ────────────────┤
                                              │
                                       EfficientNet-B0
                                              │
                                             CBAM
                                              │
                                             GeM
                                              │
                                         Classifier
                                              │
                                           P_eff
                                              │
                    ┌─────────────────────────┘
                    ↓
         Validation-based model selection
                    ↓
        Complementarity / error analysis
                    ↓
        Validation-optimized probability fusion
                    ↓
        P = alpha*P_conv + (1-alpha)*P_eff
                    ↓
              Final 38-class prediction
                    ↓
            Locked test evaluation
```

---
