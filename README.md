# QANet-PyTorch-Branchformer

Use Branchformer(https://arxiv.org/abs/2207.02971) branching logic to merge self attention and convolution output. Main changes from original QANet in lines 257 - 299 in model/QANet.py

## Introduction

Re-implement [QANet](https://arxiv.org/pdf/1804.09541.pdf) with PyTorch.
Contributions are welcomed!

## Usage

Run `python3 QANet_main.py --batch_size 32 --epochs 30 --with_cuda --use_ema ` to train model with cuda.

Run `python3 QANet_main.py --batch_size 32 --epochs 3 --with_cuda --use_ema --debug` to debug with small batches data.

## **Performance**

With ema, 8 head attention, hidden size 128, QANet_andy.model,  30 epochs, batch_size 16:

F1: **68.95**
EM: **56.43**

## Structure
QANet_main.py: code for training QANet.

trainer/QANet_trainer.py: trainer.

model/QANet_model.py: defines Modified QANet with Branchformer logic.

data_loader/SQuAD.py: SQuAD 1.1 and 2.0 data loader.

Other codes are utils or neural network common modules library.


## Acknowledge
1. The QANet structure implementation is mainly based on https://github.com/hengruo/QANet-pytorch and https://github.com/andy840314/QANet-pytorch- and https://github.com/hackiey/QAnet-pytorch.
2. For a TensorFlow implementation, please refer to https://github.com/NLPLearn/QANet.
