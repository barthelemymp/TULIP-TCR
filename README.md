# TULIP-TCR
## This repo is here to supplement the paper: TULIP — a Transformer based Unsupervised Language model for Interacting Peptides and T-cell receptors that generalizes to unseen epitopes
The accurate prediction of binding between T-cell receptors (TCR) and their cognate epitopes is key to understanding the adaptive immune response and developing immunotherapies. Current methods face two significant limitations: the shortage of comprehensive high-quality data and the bias introduced by the selection of the negative training data commonly used in the supervised learning approaches. We propose a novel method, TULIP, that addresses both limitations by leveraging incomplete data and unsupervised learning and using the transformer architecture of language models. Our model is flexible and integrates all possible data sources, regardless of their quality or completeness. We demonstrate the existence of a bias introduced by the sampling procedure used in previous supervised approaches, emphasizing the need for an unsupervised approach. TULIP recognizes the specific TCRs binding an epitope, performing well on unseen epitopes. Our model outperforms state-of-the-art models and offers a promising direction for the development of more accurate TCR epitope recognition models.


## src
The src folder contains the code for the model.


## model_weights
model weights are in the model_weights folder. You will find this the model used for the plots on HLA-A02:01 of the paper.

## data
the data folder contains the data to reproduce results of the paper.
Seenpeptides.zip: contains the data to reproduce the experiments on seen peptide. (model for this part is directly the one in model_weights)
Unseenpeptides.zip: contains the data to reproduce the experiments on unseen peptide.
RepertoireMining.zip: contains the data to reproduce the repertoire mining of neoantigen.
The largest training will be in the Seenpeptides.zip



## scripts
We give 3 scripts 
 - full_learning_new.py / run_full_leaning.sh implement the training from scratch of the model
 - finetuning.py / run_finetuning.py finetune the model on a subset
 - predict.py rank TCRs for a given epitope


## Colab:
tulip.ipynb enables playing with TULIP from colab


## Code options:
`--skipMiss`: enables the mopdel to directly learn the encoded representation of a missing sequence. Without it, the model learn the embedding of `<MIS>` and pass it to the encoder. This option is largely recommended, as it avoid having the missing sequences taking a too large importance in the training the encoders.

`--train_dir` and `--test_dir`: path to the train and test csv files.

`--modelconfig`:path to json including the config of the model.

`--save` and `--load`: path to the save the model and path to the saved model if we want to start from a pretrained model.

`--lr`L learning rate

`--weight_decay`: weight decay for the adam optimized

`--nomhc`: enables to skip the mhc. As a general rule, the mhc tokenizer is beneficial for peptides presented on HLA, for which TULIP has seen a variety of peptide.

`--num_epochs`: number of epochs

`--masking_proba`: it is a new form of regularization (not used in the paper). default is 0.0. If not null, this is the proba to randomly mask the alpha or the beta chain during training. This is made to mitigate some experimental biases on bulk vs single cell. (for example, if for a peptide we only have TCR missing their alpha chain, we would like to avoid TULIP to learn a signal between missing alpha chain and this peptide). This regularization was proven usefull when using TULIP has a geneartive model. 




## Sampling new TCR ?
Understanding how to best sample using TULIP, is still a work in progress. Sampling functions are though available.



