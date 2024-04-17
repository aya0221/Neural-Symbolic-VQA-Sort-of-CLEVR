# Neural-Symbolic-VQA-Sort-of-CLEVR

## Overview
This repository is the home of the Neural-Symbolic Visual Question Answering (VQA) system developed for the Sort-of-CLEVR dataset. Our approach combines deep learning with symbolic reasoning to effectively answer both relational and non-relational questions posed about the visual content.

### Highlights
- **Project for AI Group - Fall 2022 (Before ChatGPT Released)**
- **High Performance**: Reached 88% accuracy on relational and 99% on non-relational questions
- **Technologies Used**: Emphasizes integration of computer vision and natural language processing techniques
- **Dataset Utilized**: Employed the Sort-of-CLEVR dataset, featuring over 100k images and 1M+ questions for training and evaluation

## Execution

### Prerequisites
Ensure that you have Anaconda or Miniconda installed to manage the environment and dependencies

### Setup Environment

```bash
# Create environment from the provided environment.yml file
conda env create -f environment.yml

# Activate the newly created environment
conda activate RN3
```


### Generate Dataset
```
# Generate the Sort-of-CLEVR dataset using our script
python sort_of_clevr_generator.py
```

### Train the Model
```
# Begin training the model with the dataset
python main.py
```

# Paper Reference
The methodology and inspiration behind this project are based on the following research paper: [A simple neural network module for relational reasoning](https://arxiv.org/pdf/1706.01427.pdf)
