# Computational target inference by mining transcriptional data using a novel Siamese spectral-based graph convolutional network
This repository is a PyTorch version rewritten from https://github.com/boyuezhong/SSGCN/.

Code for "Drug target inference by mining transcriptional data using a novel graph convolutional network framework"

# System requirements
## Operating systems  requirements
This package is supported for Linux.
## Software Dependencies
The SSGCN model was implemented in the PyTorch framework (version 1.12.1) in Python 3.7.16.
## Hardware requirements
The SSGCN requires a computer with a GPU.
# Installation guide:
## Environmental setup
```
conda create -n SSGCN python=3.7
conda activate SSGCN
cd SSGCN
pip install -r requirements.txt
```

or

```
cd SSGCN
conda env create -n SSGCN -f environment.yaml
```

# Dataset
All data files are released on https://drive.google.com/drive/folders/1yHB_gE1e0cNJJeLj74ij1dtPLJiQ3fmu?usp=sharing, and the pickle form data of PC3 cell can be downloaded from https://drive.google.com/drive/folders/14odFgnwwbUhTpbExIM7L4aJWM1YYpE6-?usp=drive_link.



# Demo

```
cd SSGCN
python ./code/ssgcn_pytorch.py
```

## Expected output
The expected output can be found in ./saved_model

# Reproduction instructions
The results of benchmark can be reproduced.
# License
This code  is  licensed  under the Apache 2.0 License.

