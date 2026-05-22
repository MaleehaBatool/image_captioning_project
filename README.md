# Image Captioning with CNN-LSTM Architecture

An end-to-end deep learning pipeline that generates natural language descriptions for images. The system combines a pretrained ResNet-50 encoder for visual feature extraction with a multi-layer LSTM decoder for sequence generation, trained and evaluated on the Flickr30k dataset.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
- [Model Details](#model-details)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Requirements](#requirements)
- [Usage](#usage)
- [Results](#results)

---

## Overview

This project implements an image captioning system using a Sequence-to-Sequence (Seq2Seq) framework. Visual features are extracted from raw images using a frozen ResNet-50 backbone and projected into a shared embedding space, which initializes the hidden state of a two-layer LSTM that autoregressively generates captions token by token.

The project was developed on Kaggle and is structured to run in a GPU-enabled notebook environment.

---

## Architecture

The model consists of three components:

**ImageEncoder**
Accepts a 2048-dimensional ResNet-50 feature vector and projects it to a 512-dimensional hidden representation using a fully connected layer, followed by Batch Normalization, ReLU activation, and Dropout.

**CaptionDecoder**
A two-layer LSTM with a 256-dimensional word embedding layer. At each timestep, the decoder takes an embedded token and produces a probability distribution over the vocabulary. The encoder output initializes the hidden and cell states.

**Seq2SeqCaptioner**
Combines the encoder and decoder into a unified model. During training, teacher forcing is applied; during inference, tokens are generated autoregressively.

```
ResNet-50 (frozen) --> ImageEncoder --> LSTM Decoder --> Linear --> Vocabulary
   [2048-dim]           [512-dim]       [2-layer]                  [vocab_size]
```

Total trainable parameters: approximately 8 to 10 million depending on final vocabulary size.

---

## Dataset

**Flickr30k** — 31,783 images, each paired with five human-written reference captions.

The dataset is sourced directly from Kaggle's input directory. The pipeline dynamically locates both the image folder and the caption annotation file, supporting multiple common file formats (captions.txt, captions.csv, results.csv) and separator styles (pipe-delimited or comma-separated).

| Split      | Images  | Captions |
|------------|---------|----------|
| Train      | ~80%    | ~120,000 |
| Validation | ~10%    | ~15,000  |
| Test       | ~10%    | ~15,000  |

Splits are performed at the image level to prevent data leakage across sets.

---

## Pipeline

**Stage 1 — Feature Extraction**

ResNet-50 (pretrained on ImageNet) is truncated before its final classification layer to produce 2048-dimensional feature vectors. All images are resized to 224x224 and normalized using ImageNet statistics. Features are extracted in batches of 128 using DataParallel and persisted to disk as a pickle file (`flickr30k_features.pkl`) for reuse across training runs.

**Stage 2 — Caption Preprocessing**

Captions are lowercased, stripped of punctuation, and tokenized. A vocabulary is built from tokens appearing at least 5 times in the training set. Four special tokens are included: `<pad>`, `<start>`, `<end>`, and `<unk>`. Captions are padded or truncated to the 95th-percentile sequence length.

**Stage 3 — Training**

The model is trained for 25 epochs using Adam (lr = 3e-4) with gradient clipping at 5.0. Cross-entropy loss ignores padding tokens. A ReduceLROnPlateau scheduler halves the learning rate after 3 epochs of no validation improvement. The best checkpoint (lowest validation loss) is saved automatically.

**Stage 4 — Inference and Evaluation**

Two decoding strategies are implemented and compared against ground-truth references on the held-out test set.

---

## Model Details

| Hyperparameter     | Value   |
|--------------------|---------|
| Feature dimension  | 2048    |
| Encoder hidden dim | 512     |
| Embedding size     | 256     |
| LSTM layers        | 2       |
| Dropout            | 0.3     |
| Batch size         | 128     |
| Learning rate      | 3e-4    |
| Epochs             | 25      |
| Gradient clip      | 5.0     |
| Min token frequency| 5       |

---

## Inference

Two decoding strategies are available at inference time:

**Greedy Search**
At each step, the token with the highest probability is selected. Fast and deterministic but may produce suboptimal sequences due to locally greedy decisions.

**Beam Search**
Maintains the top-k candidate sequences (beam width = 5) at each decoding step, selecting the globally highest-scoring sequence upon completion. Produces more fluent and contextually consistent captions at the cost of additional compute.

---

## Evaluation

Predictions are evaluated against all five reference captions per image using standard captioning metrics:

| Metric       | Description                                                         |
|--------------|---------------------------------------------------------------------|
| BLEU-4       | Corpus-level 4-gram precision with brevity penalty (smoothed)       |
| Precision    | Token-level overlap between prediction and references               |
| Recall       | Coverage of reference tokens in the prediction                      |
| F1-Score     | Harmonic mean of token-level precision and recall                   |
| METEOR       | Alignment-based metric incorporating stemming and synonym matching  |

BLEU-4 and METEOR are reported as the primary metrics for comparison against published baselines.

---

## Requirements

```
torch
torchvision
numpy
pandas
scikit-learn
Pillow
tqdm
matplotlib
nltk
```

The project is designed to run on Kaggle Notebooks with GPU acceleration enabled. Ensure the Flickr30k dataset is added as a data source before execution.

---

## Usage

Run the notebook cells in order:

1. Feature extraction — produces `flickr30k_features.pkl`
2. Caption loading and preprocessing — builds vocabulary and encoded sequences
3. Dataset split — divides data into train, validation, and test sets
4. Model definition — initializes the Seq2SeqCaptioner
5. Training loop — trains for 25 epochs and saves the best checkpoint
6. Inference — runs greedy and beam search on test samples
7. Evaluation — computes BLEU-4, METEOR, and token-level metrics

The best model checkpoint is saved to `/kaggle/working/best_captioner.pth`. Sample caption outputs and the loss curve are saved as PNG files in the same directory.

---

## Results

Training and validation loss curves are saved to `loss_curve.png`. Qualitative examples comparing ground-truth captions, greedy search output, and beam search output are saved to `caption_examples.png`.

Quantitative results on the test set are printed at the end of the evaluation cell in the following format:

```
==================================================
QUANTITATIVE EVALUATION RESULTS
==================================================
BLEU-4 Score:     x.xxxx
Avg Precision:    x.xxxx
Avg Recall:       x.xxxx
Avg F1-Score:     x.xxxx
Avg METEOR:       x.xxxx
==================================================
```

---

## References

- Vinyals et al., "Show and Tell: A Neural Image Caption Generator", CVPR 2015
- He et al., "Deep Residual Learning for Image Recognition", CVPR 2016
- Young et al., "From Image Descriptions to Visual Denotations: New Similarity Metrics for Semantic Inference over Event Descriptions", TACL 2014 (Flickr30k)
- Papineni et al., "BLEU: a Method for Automatic Evaluation of Machine Translation", ACL 2002
