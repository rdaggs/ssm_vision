# MambaVision with Attention Pooling

## Overview
This repository contains modifications to the MambaVision model, introducing attention pooling for improved feature retention. Below is a summary of the steps and changes made:

---

## Baseline Model
1. **Trained MambaVision** with CIFAR-100 dataset as the baseline model.
2. The **baseline checkpoint** is located in the `checkpoints/` folder.

---

## Modifications

### Changes to `mamba_vision.py`
1. **Adjusted the model initialization logic** to support attention pooling layers.
2. **Modified the forward pass** to handle the new pooling mechanism.
3. **Original file**: `MambaVision/mamba_vision/models/mamba_vision.py`.
4. **Updated file**: `Attention_pooling/mamba_vision.py`.

### Changes to `train.py`
1. **Updated model initialization logic** to load the modified attention pooling model.
2. **Adjusted the training loop** for compatibility with the attention pooling layers.
3. **Original file**: `MambaVision/mambavision/train.py`.
4. **Updated file**: `Attention_pooling/train.py`.

---

## Running the Model
1. Use the updated `train.py` located in `Attention_pooling/`.
2. The **checkpoint for the model with attention pooling** is available in the `checkpoints/` folder.
3. Run the training script to continue or fine-tune the model:
   ```bash
   python train.py --dataset cifar100  --amp --workers 1 --epochs 100 --batch-size 16 --dataset-download --num-classes 100
