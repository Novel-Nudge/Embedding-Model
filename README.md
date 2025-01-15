# Novel Nudge Embedding Model

## Introduction

This repository contains the code for the novel nudge embedding model. This model is designed to generate embeddings for books based on the book's title and description. The resultant embeddings are then used downstream for tasks such as similarity search and clustering.

## Getting Started

### Prerequisites

- Python 3.10+
- PyTorch 2.0+
- Transformers 4.0+
- Datasets 2.0+
- TQDM 4.0+
- Pandas 2.0+

### Installation

1. Clone the repository \
   `git clone https://github.com/Novel-Nudge/Embedding-Model`

2. Install the dependencies \
   `pip install -r requirements.txt`

## TODO

- [x] Add a README.md to the src folder
- [x] Improve training loop, with checkpointing and early stopping
- [x] Add in monitoring using wandb
- [x] Add in validation
- [x] Add in evaluation steps and final evaluation
- [ ] Improve pre-processing pipeline for large datasets
