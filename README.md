# TULIP-TCR
## Attention: The sort argsort was wrong when computing the ranks. If you used the ranks and not the score please rerun your results. 
This repo implement the TULIP method for modeling the interaction between peptide and pMHC.

## src
The src folder contains the code for the model.
For now you ll find two model, a pretrained one one a large dataset of heterogenous TCRs, and a finetuned on for hla A02-01 with TCRs containing both the alpha and the beta chain.


## model_weights
model weights are in the model_weights folder

## scripts
We give 3 scripts 
 - full_learning.py / run_full_leaning.sh implement the training from scratch of the model
 - finetuning.py / run_finetuning.py finetune the model on a subset
 - predict.py rank TCRs for a given epitope

## Data split and interactive collab to come

