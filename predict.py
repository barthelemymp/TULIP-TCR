
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel, BertConfig
from sklearn.metrics import roc_auc_score

import torch

from transformers.models.encoder_decoder.configuration_encoder_decoder import EncoderDecoderConfig
from src.multiTrans import ED_BertForSequenceClassification, TCRDataset, BertLastPooler, unsupervised_auc, train_unsupervised, eval_unsupervised, MyMasking, Tulip, get_logscore

import argparse




torch.manual_seed(0)




def main():

    parser = argparse.ArgumentParser()


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
        "--output",
        default=None,
        type=str,
        help="path to save results" ,
    )
    parser.add_argument(
        "--batch_size",
        default=512,
        type=int,
        help="batch_size" ,
    )

    args = parser.parse_args()

    with open(args.modelconfig, "r") as read_file:
        print("loading hyperparameter")
        modelconfig = json.load(read_file)



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
    max_length = 114
    encoder_config = BertConfig(vocab_size = vocabsize,
                        max_position_embeddings = max_length, # this shuold be some large value
                        num_attention_heads = modelconfig["num_attn_heads"],
                        num_hidden_layers = modelconfig["num_hidden_layers"],
                        hidden_size = modelconfig["hidden_size"],
                        type_vocab_size = 1,
                        pad_token_id =  tokenizer.pad_token_id)

    encoder_config.mhc_vocab_size  =mhcvocabsize

    encoderA = BertModel(config=encoder_config)
    encoderB = BertModel(config=encoder_config)
    encoderE = BertModel(config=encoder_config)

    max_length = 50
    decoder_config = BertConfig(vocab_size = vocabsize,
                        max_position_embeddings = max_length, 
                        num_attention_heads = modelconfig["num_attn_heads"],
                        num_hidden_layers = modelconfig["num_hidden_layers"],
                        hidden_size = modelconfig["hidden_size"],
                        type_vocab_size = 1,
                        is_decoder=True, 
                        pad_token_id =  tokenizer.pad_token_id)   
    
    decoder_config.add_cross_attention=True

    decoderA = ED_BertForSequenceClassification(config=decoder_config)
    decoderA.pooler = BertLastPooler(config=decoder_config)
    decoderB = ED_BertForSequenceClassification(config=decoder_config) 
    decoderB.pooler = BertLastPooler(config=decoder_config)
    decoderE = ED_BertForSequenceClassification(config=decoder_config) 
    decoderE.pooler = BertLastPooler(config=decoder_config)
    # Define encoder decoder model
    model = Tulip(encoderA=encoderA,encoderB=encoderB,encoderE=encoderE, decoderA=decoderA, decoderB=decoderB, decoderE=decoderE)

    def count_parameters(mdl):
        return sum(p.numel() for p in mdl.parameters() if p.requires_grad)


    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_normal_(p)

    if args.load:
        checkpoint = torch.load(args.load+"/pytorch_model.bin")
        model.load_state_dict(checkpoint)
        print("loaded")
    model.to(device)
    target_peptidesFinal = pd.read_csv(test_path)["peptide"].unique()

    for target_peptide in target_peptidesFinal:
        results = pd.DataFrame(columns=["CDR3a", "CDR3b", "peptide", "rank"])
        datasetPetideSpecific= TCRDataset(test_path, tokenizer, device,target_peptide=target_peptide, mhctok=mhctok)
        print(target_peptide)
        scores = -1*np.array(get_logscore(datasetPetideSpecific, model, ignore_index =  tokenizer.pad_token_id))
        ranks = np.argsort(np.argsort(scores))
        results["CDR3a"] = datasetPetideSpecific.alpha
        results["CDR3b"] = datasetPetideSpecific.beta
        results["peptide"] = target_peptide
        results["rank"] = ranks
        results.to_csv(args.save + target_peptide+".csv")


if __name__ == "__main__":
    main()
