
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
import pandas as pd
import sys

import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate
#from data import TranslationDataset
from transformers import BertTokenizerFast, BertTokenizer
from transformers import BertModel, BertForMaskedLM, BertConfig, EncoderDecoderModel, BertLMHeadModel, AutoModelForSequenceClassification
from sklearn.metrics import roc_auc_score

import sys
import torch
import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence
import os


from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertOnlyMLMHead, SequenceClassifierOutput
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import ModelOutput

from transformers import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from transformers.models.encoder_decoder.configuration_encoder_decoder import EncoderDecoderConfig
import warnings
from torch.profiler import profile, record_function, ProfilerActivity
import wandb

import argparse
from src.multiTrans import TulipPetal, TCRDataset, BertLastPooler, unsupervised_auc, train_unsupervised, eval_unsupervised, MyMasking, Tulip, get_auc_mi

from wandb_osh.hooks import TriggerWandbSyncHook

wandb.login()


torch.manual_seed(0)




def main():
    trigger_sync = TriggerWandbSyncHook()
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--train_dir",
        default=None,
        type=str,
        required=True,
        help="The train data dir. Should contain the .csv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--test_dir",
        default=None,
        type=str,
        required=True,
        help="The test data dir. Should contain the .fasta files (or other data files) for the task.",
    )
    parser.add_argument(
        "--modelconfig",
        type=str,
        help="path to json including the config of the model" ,
    )
    parser.add_argument(
        "--load",
        default=None,
        type=str,
        help="path to the model pretrained to load" ,
    )
    parser.add_argument(
        "--save",
        default=None,
        type=str,
        help="path to save the model" ,
    )
    parser.add_argument(
        "--batch_size",
        default=512,
        type=int,
        help="batch_size" ,
    )
    parser.add_argument(
        "--masking_proba",
        default=0.0,
        type=float,
        help="masking_proba" ,
    )
    parser.add_argument(
        "--num_epochs",
        default=300,
        type=int,
        help="numbers of epochs" ,
    )
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="weight decay" ,
    )
    parser.add_argument(
        "--lr",
        default=0.0001,
        type=float,
        help="learning rate" ,
    )


    parser.add_argument("--freeze", action="store_true", help="Whether to run training.")
    parser.add_argument("--skipMiss", action="store_true", help="Whether to run training.")

    args = parser.parse_args()

    with open(args.modelconfig, "r") as read_file:
        print("loading hyperparameter")
        modelconfig = json.load(read_file)








    wandb.login()
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)


    test_path = args.test_dir
    train_path = args.train_dir


    tokenizer = AutoTokenizer.from_pretrained("aatok/")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<PAD>'})

    if tokenizer.sep_token is None:
        tokenizer.add_special_tokens({'sep_token': '<MIS>'})
        
    if tokenizer.cls_token is None:
        tokenizer.add_special_tokens({'cls_token': '<CLS>'})

    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({'eos_token': '<EOS>'})

    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({'mask_token': '<MASK>'})



    from tokenizers.processors import TemplateProcessing
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single="<CLS> $A <EOS>",
        pair="<CLS> $A <MIS> $B:1 <EOS>:1",
        special_tokens=[
            ("<EOS>", 2),
            ("<CLS>", 3),
            ("<MIS>", 4),
        ],
    )

    mhctok = AutoTokenizer.from_pretrained("mhctok/")

    vocabsize = len(tokenizer._tokenizer.get_vocab())
    mhcvocabsize = len(mhctok._tokenizer.get_vocab())
    print("Loading models ..")

    max_length = 50
    encoder_config_pep = BertConfig(vocab_size = vocabsize,
                        max_position_embeddings = max_length, # this shuold be some large value
                        num_attention_heads = modelconfig["num_attn_heads_encoder_pep"],
                        num_hidden_layers = modelconfig["num_hidden_layers_encoder_pep"],
                        hidden_size = modelconfig["hidden_size"],
                        type_vocab_size = 1,
                        pad_token_id =  tokenizer.pad_token_id)
    
    encoder_config_cdr = BertConfig(vocab_size = vocabsize,
                        max_position_embeddings = max_length, # this shuold be some large value
                        num_attention_heads = modelconfig["num_attn_heads_encoder_cdr"],
                        num_hidden_layers = modelconfig["num_hidden_layers_encoder_cdr"],
                        hidden_size = modelconfig["hidden_size"],
                        type_vocab_size = 1,
                        pad_token_id =  tokenizer.pad_token_id)

    encoder_config_pep.mhc_vocab_size  = mhcvocabsize
    encoder_config_cdr.mhc_vocab_size  = mhcvocabsize

    encoderA = BertModel(config=encoder_config_cdr)
    encoderB = BertModel(config=encoder_config_cdr)
    encoderE = BertModel(config=encoder_config_pep)

    max_length = 50
    decoder_config_pep = BertConfig(vocab_size = vocabsize,
                        max_position_embeddings = max_length, # this shuold be some large value
                        num_attention_heads = modelconfig["num_attn_heads_decoder_pep"],
                        num_hidden_layers = modelconfig["num_hidden_layers_decoder_pep"],
                        hidden_size = modelconfig["hidden_size"],
                        type_vocab_size = 1,
                        is_decoder=True, 
                        pad_token_id =  tokenizer.pad_token_id)    # Very Important
    
    decoder_config_cdr = BertConfig(vocab_size = vocabsize,
                        max_position_embeddings = max_length, # this shuold be some large value
                        num_attention_heads = modelconfig["num_attn_heads_decoder_cdr"],
                        num_hidden_layers = modelconfig["num_hidden_layers_decoder_cdr"],
                        hidden_size = modelconfig["hidden_size"],
                        type_vocab_size = 1,
                        is_decoder=True, 
                        pad_token_id =  tokenizer.pad_token_id)    # Very Important
    
    decoder_config_cdr.add_cross_attention=True
    decoder_config_pep.add_cross_attention=True

    decoderA = TulipPetal(config=decoder_config_cdr) #BertForMaskedLM
    decoderA.pooler = BertLastPooler(config=decoder_config_cdr)
    decoderB = TulipPetal(config=decoder_config_cdr) #BertForMaskedLM
    decoderB.pooler = BertLastPooler(config=decoder_config_cdr)
    decoderE = TulipPetal(config=decoder_config_pep) #BertForMaskedLM
    decoderE.pooler = BertLastPooler(config=decoder_config_pep)


    # Define encoder decoder model
    model = Tulip(encoderA=encoderA,encoderB=encoderB,encoderE=encoderE, decoderA=decoderA, decoderB=decoderB, decoderE=decoderE)
    
    if args.skipMiss:
        model.skipMiss=True
    else:
        model.skipMiss=False

    def count_parameters(mdl):
        return sum(p.numel() for p in mdl.parameters() if p.requires_grad)

    print(f'The model has {count_parameters(model):,} trainable parameters')

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_normal_(p)

    if args.load:
        checkpoint = torch.load(args.load+"/pytorch_model.bin")
        model.load_state_dict(checkpoint)
        print("loaded")


    model.to(device)

    if args.freeze:
        for param in model.mhc_embeddings.parameters(): #params have requires_grad=True by default
            param.requires_grad = False
        for param in model.encoderA.parameters(): #params have requires_grad=True by default
            param.requires_grad = False
        for param in model.encoderB.parameters(): #params have requires_grad=True by default
            param.requires_grad = False
        for param in model.encoderE.parameters(): #params have requires_grad=True by default
            param.requires_grad = False

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print("pti",tokenizer.pad_token_id)
    criterion = nn.NLLLoss(ignore_index=tokenizer.pad_token_id,  reduction='sum')


    datasetTrainFull = TCRDataset(train_path, tokenizer, device, mhctok=mhctok)
    datasetTrainFull.set_chain_masking_proba(proba=args.masking_proba)
    train_dataloaderFull = torch.utils.data.DataLoader(dataset=datasetTrainFull, batch_size=args.batch_size, shuffle=True, collate_fn=datasetTrainFull.all2allmhc_collate_function) 
    datasetValidFinal = TCRDataset(test_path, tokenizer, device, mhctok=mhctok)
    valid_dataloaderFinal = torch.utils.data.DataLoader(dataset=datasetValidFinal, batch_size=args.batch_size, shuffle=True, collate_fn=datasetValidFinal.all2allmhc_collate_function) 
    datasetValidFinal_true = TCRDataset(test_path, tokenizer, device,target_binder=True, mhctok=mhctok)
    valid_dataloaderFinal_true = torch.utils.data.DataLoader(dataset=datasetValidFinal_true, batch_size=args.batch_size, shuffle=True, collate_fn=datasetValidFinal.all2allmhc_collate_function) 


    masker = MyMasking(tokenizer, mlm_probability = 0.15)


    config_dict = {
        'batch_size':  args.batch_size,
        'dropout':  0.1,
        'lr':  args.lr,
        'weight_decay': args.weight_decay,
        'skipMiss': args.skipMiss,
        'masking_proba':args.masking_proba,
        'starting_point': args.load,
        'freeze':args.freeze,
    }
    config_dict.update(modelconfig)

    wandb.init(project="finetune_reviews", entity="barthelemymp", config=config_dict)
    trigger_sync() 
    wandb.config.update(config_dict) 
    trigger_sync()

    target_peptidesFinal = pd.read_csv(test_path)["peptide"].unique()




    target_peptidesFinal = pd.read_csv(test_path)["peptide"].value_counts().index
    target_peptidesFinal_top = pd.read_csv(test_path)["peptide"].value_counts().index[:20]



    for epoch in range(0, args.num_epochs+1):
        if epoch%20==0:
            aucelist = []
            aucalist = []
            aucblist = []
            aucmialist = []
            aucmiblist = []
            for target_peptide in target_peptidesFinal:
                datasetPetideSpecific= TCRDataset(test_path, tokenizer, device, target_peptide=target_peptide, mhctok=mhctok)
                dataloaderPetideSpecific = torch.utils.data.DataLoader(dataset=datasetPetideSpecific, batch_size=1, shuffle=True, collate_fn=datasetValidFinal.all2allmhc_collate_function) 
                print(target_peptide)
                sys.stdout.flush()
                auca, aucb, auce = unsupervised_auc(model, dataloaderPetideSpecific, tokenizer.pad_token_id)
                aucami, aucbmi = get_auc_mi(model, datasetPetideSpecific, mask_mhc=True, mask_peptide=True, mask_paired=False)
                wandb.log({target_peptide+"_a":auca, target_peptide+"_b":aucb,target_peptide+"_mia":aucami, target_peptide+"_mib":aucbmi,target_peptide+"_e":auce, "epochT":epoch})
                

                aucelist.append(auce)
                aucalist.append(auca)
                aucblist.append(aucb)
                aucmialist.append(aucami)
                aucmiblist.append(aucbmi)
            wandb.log({"avg_e":np.mean(aucelist), "avg_a":np.mean(aucalist),"avg_b":np.mean(aucblist), "avg_mia":np.mean(aucmialist),"avg_mib":np.mean(aucmiblist),"epochT":epoch})
            trigger_sync() 

        print("Starting epoch", epoch+1, file=sys.stdout)
        sys.stdout.flush()
        epoch_lm_lossA, epoch_lm_lossB, epoch_lm_lossE, epoch_mlm_lossA, epoch_mlm_lossB, epoch_mlm_lossE = train_unsupervised(model, optimizer, masker, train_dataloaderFull, criterion)
        print(epoch_lm_lossA, epoch_lm_lossB, epoch_lm_lossE, epoch_mlm_lossA, epoch_mlm_lossB, epoch_mlm_lossE,file=sys.stdout)
        wandb.log({"epoch_lm_lossAu": epoch_lm_lossA, "epoch_lm_lossBu":epoch_lm_lossB ,"epoch_lm_lossEu":epoch_lm_lossE ,"epoch_mlm_lossAu":epoch_mlm_lossA ,"epoch_mlm_lossBu":epoch_mlm_lossB ,"epoch_mlm_lossEu":epoch_mlm_lossE, "epochT":epoch})
        trigger_sync()
        if epoch%20==0:
            with  torch.no_grad():
                epoch_lm_lossA, epoch_lm_lossB, epoch_lm_lossE, epoch_mlm_lossA, epoch_mlm_lossB, epoch_mlm_lossE = eval_unsupervised(model, masker, valid_dataloaderFinal_true, criterion)
                print(epoch_lm_lossA, epoch_lm_lossB, epoch_lm_lossE, epoch_mlm_lossA, epoch_mlm_lossB, epoch_mlm_lossE,file=sys.stdout)
                sys.stdout.flush()
                wandb.log({"epoch_lm_lossAu_val": epoch_lm_lossA, "epoch_lm_lossBu_val":epoch_lm_lossB ,"epoch_lm_lossEu_val":epoch_lm_lossE ,"epoch_mlm_lossAu_val":epoch_mlm_lossA ,"epoch_mlm_lossBu_val":epoch_mlm_lossB ,"epoch_mlm_lossEu_val":epoch_mlm_lossE, "epochT":epoch})
            
        if epoch%10==0:
            if args.save:
                print('saving model at ', args.save + str(epoch))
                model.save_pretrained(args.save + str(epoch))


if __name__ == "__main__":
    main()
