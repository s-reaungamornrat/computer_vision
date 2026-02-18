In the field of video classification, **UCF101** remains a staple benchmark. However, because it is a relatively small dataset by modern standards (13,320 videos), results can vary significantly based on whether the model was trained from scratch or fine-tuned from a larger dataset like Kinetics-400.

For **Fold 1** specifically (which is the most commonly reported split in papers), the accuracies generally fall into three performance "tiers":

### 1. State-of-the-Art (SOTA) Tiers

Modern models, particularly those leveraging **Vision Transformers (ViT)** and massive pre-training, have pushed UCF101 performance near saturation.

| Model Category | Top-1 Accuracy | Top-5 Accuracy | Note |
| --- | --- | --- | --- |
| **High-End Transformers** | **98.5% – 99.7%** | **~99.9%** | Models like *VideoMAE V2* or *UniFormerV2* pre-trained on massive datasets. |
| **Standard 3D CNNs** | **94.0% – 98.0%** | **~98.5%** | Models like *I3D* or *R(2+1)D* (often pre-trained on Kinetics). |
| **Legacy/Baseline** | **88.0% – 93.0%** | **~95.0%** | Original *Two-Stream* networks or *C3D*. |

### 2. Training from Scratch vs. Pre-training

If you are looking for what a researcher gets when training a model *only* on UCF101 Fold 1 (without using ImageNet or Kinetics weights), the numbers drop dramatically:

* **Training from Scratch:** Top-1 accuracy usually hovers between **40% and 60%**. The dataset is simply too small to converge modern deep architectures effectively without a "warm start" from pre-trained weights.
* **Fine-tuned (Standard Practice):** Most researchers report results after pre-training on **Kinetics-400**. In this scenario, you should expect a Top-1 of **>90%** and a Top-5 of **>97%**.

### 3. Why Top-5 is so high

On UCF101, the Top-5 accuracy is often exceptionally high (approaching 99% for many models) because:

* **Class Distinctness:** Many classes are visually distinct (e.g., "Biking" vs. "Haircut"). If the model misses the specific action, the correct label is almost always in the next few guesses.
* **Data Leakage/Overlap:** Since UCF101 is organized into groups (videos in a group often share the same actor or background), models can easily "memorize" certain visual cues that apply across multiple videos in the test set.


--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

In video classification, there is a distinct gap between **Clip-level** (a single short snippet) and **Video-level** (averaging predictions across the whole video) accuracy. On a dataset like **UCF101 Fold 1**, these metrics also shift dramatically depending on whether you start from a blank slate or use a "warmed-up" model.

### 1. Training From Scratch (No Pre-training)

Training modern 3D architectures (like I3D, R(2+1)D, or SlowFast) purely on UCF101 is difficult because the dataset is too small to prevent overfitting.

* **Video-level (Top-1):** Generally **40% – 60%**. Some optimized implementations of SlowFast or R3D-18 can push this to **~70%** with heavy augmentation and long training (500+ epochs), but this is not the standard baseline.
* **Clip-level (Top-1):** Usually **5% – 10% lower** than video-level (~35% – 50%).
* **Top-5 Accuracy:** Often falls in the **65% – 85%** range.

### 2. Fine-tuning a Pre-trained Model

This is the standard research protocol. Models are typically pre-trained on **ImageNet** (2D) or **Kinetics-400** (3D) before being "tuned" on UCF101.

| Metric | Clip-level (Single Clip) | Video-level (Full Video) |
| --- | --- | --- |
| **Top-1 Accuracy** | **85% – 92%** | **94% – 98%** |
| **Top-5 Accuracy** | **~96%** | **98.5% – 99.7%** |

*Note: The **Video-level** accuracy is higher because it uses a "consensus" method (e.g., 10-clip testing), where the model looks at 10 different parts of the video and averages the scores, which smooths out errors from any single "noisy" clip.*

---

### Comparison Table: Why Pre-training Matters

Using a classic model like **I3D** as a benchmark for Fold 1:

| Training Strategy | Top-1 (Video) | Top-5 (Video) |
| --- | --- | --- |
| **From Scratch** | ~48.8% | ~76.0% |
| **Pre-trained (Kinetics-400)** | **95.6%** | **~99.0%** |
| **State-of-the-Art (ViT/MAE)** | **99.0%+** | **~99.9%** |

### Why the gap?

1. **Temporal Breadth:** A single **Clip** (usually 16–32 frames) might only show a person's face or a background, while the **Video** level aggregates these, ensuring the "action" (like a golf swing) is eventually seen.
2. **Dataset Saturation:** UCF101 is now considered "solved" when using pre-training. Researchers mostly use it today to test **Transfer Learning** efficiency rather than raw architecture power.

**Would you like me to provide the specific training parameters (learning rate, epochs, etc.) used to achieve these pre-trained results?**