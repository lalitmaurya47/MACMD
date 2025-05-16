# MACMD: Multi-dilated Contextual Attention and Channel Mixer Decoding for Medical Image Segmentation 

This repository contains the **testing code** for our **BMVC 2025 submission**, which introduces a novel decoder design called **MACMD** (Multi-dilated Contextual Attention and Channel Mixer Decoding for Medical Image Segmentation ). 



> 🚧 **Note**: The training code and full documentation will be made available soon. Please stay tuned.

## 🧪 Testing Instructions

This code is designed to **evaluate pretrained MACMD models** on multiple benchmark datasets such as:

- BUSI
- ISIC 2017
- Synapse

Download the Synapse dataset from [here](https://drive.google.com/uc?export=download&id=18I9JHH_i0uuEDg-N6d7bfMdf7Ut6bhBi).

### Pretrained model:
Pretrained weights for the encoder can be downloaded at PVTv2 model from their original source [Google Drive](https://drive.google.com/drive/folders/1Eu8v9vMRvt-dyCH0XSV2i77lAd62nPXV?usp=sharing)/[PVT GitHub](https://github.com/whai362/PVT/releases/tag/v2), and then put it in the './pretrained_pth/pvt/' folder for initialization.

Checkpoints for PVT-B2-MACMD on the Synapse data can be downloaded at [link](https://drive.usercontent.google.com/download?id=132wfiwJYy1BGdl4b1mjUanG7LzwDrIj2&export=download&authuser=0)

### Testing:
Run the jupyter script test.ipynb for the Synapse dataset. Make sure to configure your dataset path and other parameters in configs/config_setting.py.

### ✅ Requirements

- Python ≥ 3.7
- PyTorch ≥ 1.10
- NVIDIA GPUs with CUDA support
- Required Python packages (install via pip):

```bash
pip install -r requirements.txt

Make sure to configure your dataset path and other parameters in configs/config_setting.py.