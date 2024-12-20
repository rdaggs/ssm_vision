# MambaVision with Attention Pooling

## Overview
This repository contains modifications to the MambaVision model, introducing attention pooling for improved feature retention. Below is a summary of the steps and changes made:

---

## Baseline Model
1. **Trained MambaVision** with CIFAR-100 dataset as the baseline model.
2. **The Baseline Checkpoint** is given below.

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
2. The **checkpoint for the model with attention pooling** is available below.
3. Run the training script to continue or fine-tune the model:
   ```bash
   python train.py --dataset cifar100  --amp --workers 1 --epochs 100 --batch-size 16 --dataset-download --num-classes 100 --resume MambaVision/output/train/exp//Attention-pooling-check.pth.tar

---

### Model Checkpoints

The following are the model checkpoints for the Baseline MambaVision model and the Attention Pooling modification. These files are hosted on Google Drive and can only be accessed with NYU IDs. Please ensure you are logged in with your NYU credentials to access these links.

- **Baseline MambaVision Checkpoint**  
  File: `Basemodel-134.pth.tar`  
   [Download here](https://drive.google.com/file/d/1iB8l07U4bD19PCRIdoa7wdt69B9iBxVV/view?usp=sharing)

- **Attention Pooling Checkpoint**  
  File: `Attention-pooling-check.pth.tar`  
  [Download here](https://drive.google.com/file/d/1VzMPsbY02L2pT9_AEwn7BCZlEG0KQ8It/view?usp=sharing)

---

### Instructions to Use

#### 1. Download the Checkpoints
- Ensure you are logged in with your NYU ID.
- Download the checkpoint files using the links provided above.

#### 2. Place in the Checkpoints Directory
- Place the downloaded `.pth.tar` files in the `MambaVision/output/train/exp/` directory of the project.

#### 3. Run the Model
- Use the `--resume` flag to load the respective checkpoint when running the training or evaluation script.

```bash
python train.py --dataset cifar100  --amp --workers 1 --epochs 100 --batch-size 16 --dataset-download --num-classes 100 --resume MambaVision/output/train/exp//Attention-pooling-check.pth.tar

