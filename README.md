# SNHE
This repo is for source code of paper "Self-supervised Nodes-Hyperedges Embedding for Heterogeneous Information Network Learning". This code is referenced from [https://raw.githubusercontent.com/liun-online/HeCo](https://github.com/liun-online/HeCo), thanks to the author's contribution.

## Environment Settings
> python==3.8.5 \
> scipy==1.6.2 \
> torch==1.12.0 \
> torch-geometric==2.2.0 \
> PyGCL==0.1.1 \
> numpy==1.19.2 \
> scikit_learn==1.0 \
GPU: GeForce RTX 3090 
## Dataset
The data obtained from [link](https://github.com/liun-online/HeCo/tree/main/data) is approximate.


## Usage
Fisrt, go into ./code, you can use the following commend to generate pre-training files
> python pretrain.py

and then you can use the following commend to run our model: 
> python main.py acm --gpu=0

Here, "acm" can be replaced by "dblp", "aminer" or "imdb".

## Contcat
This code has not been thoroughly verified and is only for learning and communication purposes. Please feel free to raise any questions or suggest better solutions by contacting limengran1998@163.com.
