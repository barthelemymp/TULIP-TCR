# TULIP-TCR
## This is a new version of the repo, goung with the updated version of the paper.

## src
The src folder contains the code for the model.


## model_weights
model weights are in the model_weights folder. You will find this the model used for the plots on HLA-A02:01 of the paper.

## data
the data folder contains the data to reproduce results of the paper.
The largest training will be in the Seen.zip



## scripts
We give 3 scripts 
 - full_learning_new.py / run_full_leaning.sh implement the training from scratch of the model
 - finetuning.py / run_finetuning.py finetune the model on a subset
 - predict.py rank TCRs for a given epitope


## Colab:
tulip.ipynb enables playing with TULIP from colab

## Sampling new TCR ?
Understanding how to best sample using TULIP, is still a work in progress. Sampling functions are though available.
