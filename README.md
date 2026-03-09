# RGait-USDA
RGait-USDA: A Class-Aware Unsupervised Subdomain Adaptation Framework for Covariate-Robust Radar Gait Recognition

## Requirement
* python 3.9
* numpy==1.24.0
* Pillow==11.1.0
* torch==2.4.0
* torchvision==0.19.0

## Usage
1. You can download the dataset to the dataset folder in this directory, then modify the root_path variable in the code to your actual path.
2. You can replace model_pre.pth with the initialization model trained by yourself (or adjust the code to point to your own trained initialization model file instead of model_pre.pth).
3. You can adjust the 'src', 'tgt', and 'test' parameters in 'main.py' to implement different transfer learning tasks. Note that 'tgt' and 'test' refer to the same dataset but adopt different input formats; you also need to modify the number of samples in the 'creat_bank' custom function within 'main.py' to match the creation of the feature bank.
4. Run python `main.py`.
