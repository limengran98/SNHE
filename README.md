# SNHE
This repo is for source code of paper "Self-supervised Nodes-Hyperedges Embedding for Heterogeneous Information Network Learning". This code is referenced from https://raw.githubusercontent.com/liun-online/HeCo, thanks to the author's contribution.

## Environment Settings
> python==3.8.5 \
> scipy==1.5.4 \
> torch==1.7.0 \
> numpy==1.19.2 \
> scikit_learn==0.24.2
GPU: GeForce RTX 3090 
## Dataset
The data set is available at the following link


## Usage
Fisrt, go into ./code, and then you can use the following commend to run our model: 
> python main.py acm --gpu=0

Here, "acm" can be replaced by "dblp", "aminer" or "imdb".


