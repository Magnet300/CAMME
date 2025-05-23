# CAMME: Adaptive Deepfake Image Detection with Multi-Modal Cross-Attention

## Overview
![CAMME Architecture](figs/CAMME_framework.png)

The proposed framework for deepfake image detection utilizes an Image Encoder, Text Encoder, and Frequency Encoder to produce respective embeddings. These embeddings are processed as tokens through a Transformer Block with a self-attention mechanism, enabling cross-attention across visual, textual, and frequency domains. The aggregated embedding from these tokens is used for classification.

## Results
### Natural Scene Datasets
![Evaluation Results on Natural Scene Datasets](figs/Table_2.png)

### Face Datasets
![Evaluation Results on Face Datasets](figs/Table_3.png)

The tables display intra-domain performance and Inter-domain Average (IA), calculated from inter-domain transfer tasks to measure model performance on unseen domains.

## Dataset

Download the dataset from Google Drive:  
[**CAMME_dataset Link**](https://drive.google.com/drive/folders/1lMpD-EjDfWFpbhcPT9KSKBYzzgXiTLhC?usp=sharing)


# Environment Setup

To set up the environment, follow these steps:

```bash
conda env create -f environment.yaml
conda activate CAMME


# Train the model
python train.py --train_real_dir /path/to/real/train/real --train_fake_dir /path/to/fake/train/fake --val_real_dir /path/to/real/val/real --val_fake_dir /path/to/fake/val/fake

# Test the model
python test.py --model_dir /path/to/trained_model.pth --test_real_dir /path/to/real/test/real --test_fake_dir /path/to/fake/test/fake 


