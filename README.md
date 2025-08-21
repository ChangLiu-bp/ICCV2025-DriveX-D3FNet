# D3FNet: Differential Attention Fusion Network for Fine-Grained Road Extraction

Official implementation of our ICCV 2025 paper:  
**"D3FNet: A Dilated Dual-Stream Differential Attention Fusion Network for Fine-Grained Road Structure Extraction in Remote Perception Systems"**  

<p align="center">
  <img src="docs/framework.png" width="600"/>
</p>

---

## 🔍 Abstract

Extracting narrow roads from high-resolution remote sensing imagery remains a significant challenge due to their limited width, fragmented topology, and frequent occlusions. To address these issues, we propose **D3FNet**, a *Dilated Dual-Stream Differential Attention Fusion Network* designed for fine-grained road structure segmentation in remote perception systems. Built upon the encoder–decoder backbone of D-LinkNet, D3FNet introduces three key innovations:

1. **Differential Attention Dilation Extraction (DADE)** module to enhance subtle road features while suppressing background noise;
2. **Dual-stream Decoding Fusion Mechanism (DDFM)** that integrates original and attention-modulated features for better spatial precision and semantic context;
3. **Multi-scale dilation strategy (rates 1, 3, 5, 9)** to reduce gridding artifacts and improve continuity in narrow road prediction.

Extensive experiments on **DeepGlobe** and **CHN6-CUG** benchmarks show that D3FNet achieves superior IoU and recall on challenging road regions, outperforming state-of-the-art baselines.  

---

## ✨ Key Contributions

- 🔹 We introduce **D3FNet**, combining differential attention and dual-stream decoding into the D-LinkNet framework.  
- 🔹 A novel **DADE module** improves feature discrimination and suppresses cluttered backgrounds.  
- 🔹 A **DDFM structure** preserves continuity in occluded/narrow roads.  
- 🔹 State-of-the-art performance on **DeepGlobe** and **CHN6-CUG**, validating robustness in cooperative driving scenarios.  

---

## 📂 Repository Structure


TOTO


# DeepGlobe-Road-Extraction-Challenge
Code for the 1st place solution in [DeepGlobe Road Extraction Challenge](https://competitions.codalab.org/competitions/18467).

# Requirements

- Cuda 8.0
- Python 2.7
- Pytorch 0.2.0
- cv2

# Usage

### Data
Place '*train*', '*valid*' and '*test*' data folders in the '*dataset*' folder.

Data is from [DeepGlobe Road Extraction Challenge](https://competitions.codalab.org/competitions/18467#participate-get_starting_kit). You should sign in first to get the data.

### Train
- Run `python train_lr.py` to train the default D-LinkNet34.
Remember to choose the right model from Diffdlinknet_v6.py, which is Dlinknet34.
The input image must be a three-channel RGB image and have a square shape.

### Predict
- Run `Validation.py` to predict on the default D-LinkNet34.
Remember to choose the right model from Diffdlinknet_v6.py, which is Dlinknet34.
Then should change source dataset and load the weight that have been trained.

### Download trained D-LinkNet34
- [Dropbox](https://www.dropbox.com/sh/h62vr320eiy57tt/AAB5Tm43-efmtYzW_GFyUCfma?dl=0)
- [百度网盘](https://pan.baidu.com/s/1wqyOEkw5o0bzbuj7gBMesQ)
