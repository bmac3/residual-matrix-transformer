# Residual Matrix Transformers

This respository contains code for the paper "Residual Matrix Transformers: Scaling the Size of the Residual Stream". The paper can be found here (https://arxiv.org/abs/2506.22696) and a sort video can be found on the ICML poster page (https://icml.cc/virtual/2025/poster/45278).

![ICML_2025_poster](https://github.com/user-attachments/assets/2043927a-be19-470b-bf3f-2d047cc31945)


# Abstract

The residual stream acts as a memory bus where transformer layers both store and access features. We consider changing the mechanism for retrieving and storing information in the residual stream, and replace the residual stream of the transformer with an outer product memory matrix. We call this model the Residual Matrix Transformer (RMT). We find that the RMT enjoys a number of attractive properties: 1) the size of the residual stream can be scaled independently of compute and model size, improving performance, 2) the RMT can achieve the same loss as the transformer with 58% fewer FLOPS, 25% fewer parameters, and 41% fewer training tokens tokens, and 3) the RMT outperforms the transformer on downstream evaluations. We theoretically analyze the transformer and the RMT, and show that the RMT allows for more efficient scaling of the residual stream, as well as improved variance propagation properties.

# Installation

Simply running `pip install -r requirements.txt` will install all required depencencies and project code.

# Running Code

Running `dvc repro` from the command line will run a pipeline that downloads/tokenizes data and trains a transformer and rmt model. The results are logged using tensorboard. The training setup can be tweeked by changing relavent variables in `params.yaml`.
