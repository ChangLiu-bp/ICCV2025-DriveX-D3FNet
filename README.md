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
