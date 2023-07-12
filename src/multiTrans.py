
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
from typing import List, Optional, Tuple, Union, Any
from transformers.modeling_outputs import ModelOutput

from transformers import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from transformers.models.encoder_decoder.configuration_encoder_decoder import EncoderDecoderConfig
import warnings

# model = AutoModelForCausalLM.from_pretrained("lightonai/RITA_s", trust_remote_code=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
# rita_gen = pipeline('text-generation', model=model, tokenizer=tokenizer)
# sequences = rita_gen("MAB", max_length=20, do_sample=True, top_k=950, repetition_penalty=1.2, 
#                      num_return_sequences=2, eos_token_id=2)
# for seq in sequences:
#     print(f"seq: {seq['generated_text'].replace(' ', '')}")



class TCRDataset(data.Dataset):

    def __init__(self, csv_file, tokenizer, device, target_binder=None, target_peptide=None, excluded_peptide=None, mhctok=None):#, alpha_maxlength, beta_maxlength, epitope_maxlength):
        self.device=device
        self.tokenizer = tokenizer
#         self.alpha_maxlength = alpha_maxlength
#         self.beta_maxlength = beta_maxlength
#         self.epitope_maxlength = epitope_maxlength

        print("Loading and Tokenizing the data ...")
#         self.encoded_alpha = []
#         self.encoded_beta = []
#         self.encoded_epitope = []
#         self.HLA = []
#         self.binder = []
        # Read the EN lines
        num_inp_lines = 0

        df = pd.read_csv(csv_file)
        
        if target_binder:
            df = df[df["binder"]==1]

        if target_peptide:
            df = df[df["peptide"].apply(lambda x: x in target_peptide)]
            # df = df[df["peptide"]==target_peptide]
        if excluded_peptide:
            print("exluded", excluded_peptide)
            iii = df["peptide"].apply(lambda x: x in excluded_peptide)
            df = df[~iii]


        self.alpha = list(df["CDR3a"])
        self.beta = list(df["CDR3b"])
        self.peptide = list(df["peptide"])
        self.binder = list(df["binder"])
        if mhctok:
            self.mhctok = mhctok
            self.MHC = list(df["MHC"])
        self.df = df
        self.reweight=False
        #self.encoded_alpha = list(df.apply(lambda x: self.tokenizer.encode(x["CDR3a"],padding="max_length", add_special_tokens=True, max_length=self.alpha_maxlength), axis=1))
        #self.encoded_beta = list(df.apply(lambda x: self.tokenizer.encode(x["CDR3b"],padding="max_length", add_special_tokens=True, max_length=self.beta_maxlength), axis=1))
        #self.encoded_alphabeta =  self.tokenizer(list(df["CDR3a"]),list(df["CDR3b"]),padding="longest", add_special_tokens=True)
        #self.encoded_epitope = self.tokenizer(list(df["peptide"]),padding="longest", add_special_tokens=True)
       
        # with open(inp_file, "r") as ef:
        #     for line in ef:
        #         enc = self.tokenizer.encode(line.strip(), add_special_tokens=True, max_length=self.inp_maxlength)
        #         self.encoded_inp.append(torch.tensor(enc))
        #         num_inp_lines += 1

        # # read the DE lines
        # num_targ_lines = 0
        # with open(targ_file, "r") as df:
        #     for line in df:
        #         enc = self.targ_tokenizer.encode(line.strip(), add_special_tokens=True, max_length=self.targ_maxlength)
        #         self.encoded_targ.append(torch.tensor(enc))
        #         num_targ_lines += 1

        # assert (num_inp_lines==num_targ_lines), "Mismatch in EN and DE lines"
        # print("Read", num_inp_lines, "lines from EN and DE files.")

    def __getitem__(self, offset):
#         alpha = self.encoded_alpha[offset]
#         beta = self.encoded_beta[offset]
#         alphabeta = self.encoded_alphabeta[offset]
#         epitope = self.encoded_epitope[offset]
        alpha = self.alpha[offset]
        beta = self.beta[offset]
        peptide = self.peptide[offset]
        binder = self.binder[offset]
        if self.mhctok:
            mhc = self.MHC[offset]
            if self.reweight:
                w = self.weights[offset]
                return alpha, beta, peptide, binder, mhc, w

            return alpha, beta, peptide, binder, mhc


        return alpha, beta, peptide, binder

    def __len__(self):
        return len(self.peptide)

    def set_reweight(self,alpha):
        freq = self.df["peptide"].value_counts()/self.df["peptide"].value_counts().sum()
        # df.apply(lambda x: freq[x["peptide"]], 1)
        alpha = alpha
        freq = alpha*freq + (1-alpha)/len(self.df["peptide"].value_counts())
        self.weights = (1/torch.tensor(list(self.df.apply(lambda x: freq[x["peptide"]],1 ))))/len(self.df["peptide"].value_counts())
        self.reweight = True


    def ab2e_collate_function(self, batch):

        (alpha, beta, peptide, binder) = zip(*batch)
        peptide = self.tokenizer(list(peptide),padding="longest", add_special_tokens=True)
        peptide = {k: torch.tensor(v).to(self.device) for k, v in peptide.items()}#default_collate(peptide)
        
        alphabeta = self.tokenizer(list(alpha), list(beta), padding="longest", add_special_tokens=True)
        alphabeta = {k: torch.tensor(v).to(self.device) for k, v in alphabeta.items()}
        
        binder =  default_collate(binder).to(self.device)


        return peptide, alphabeta, binder

    def all2all_collate_function(self, batch):

        (alpha, beta, peptide, binder) = zip(*batch)

        peptide = self.tokenizer(list(peptide),padding="longest", add_special_tokens=True)
#         print(peptide["input_ids"])
#         print("coucou")
#         print(torch.tensor(peptide["input_ids"]))
        
        peptide = {k: torch.tensor(v).to(self.device) for k, v in peptide.items()}#default_collate(peptide)
        
        beta = self.tokenizer(list(beta),  padding="longest", add_special_tokens=True)
        beta = {k: torch.tensor(v).to(self.device) for k, v in beta.items()}

        alpha = self.tokenizer(list(alpha), padding="longest", add_special_tokens=True)
        alpha = {k: torch.tensor(v).to(self.device) for k, v in alpha.items()}
        
        binder =  default_collate(binder).to(self.device)


        return peptide, alpha, beta, binder


    def all2allmhc_collate_function(self, batch):

        if self.reweight:
            (alpha, beta, peptide, binder, mhc, weight) = zip(*batch)
        else:
            (alpha, beta, peptide, binder, mhc) = zip(*batch)

        peptide = self.tokenizer(list(peptide),padding="longest", add_special_tokens=True)
#         print(peptide["input_ids"])
#         print("coucou")
#         print(torch.tensor(peptide["input_ids"]))
        
        peptide = {k: torch.tensor(v).to(self.device) for k, v in peptide.items()}#default_collate(peptide)
        
        beta = self.tokenizer(list(beta),  padding="longest", add_special_tokens=True)
        beta = {k: torch.tensor(v).to(self.device) for k, v in beta.items()}

        alpha = self.tokenizer(list(alpha), padding="longest", add_special_tokens=True)
        alpha = {k: torch.tensor(v).to(self.device) for k, v in alpha.items()}
        
        binder =  default_collate(binder).to(self.device)
        mhc = self.mhctok(list(mhc))#default_collate(self.mhctok(list(mhc))['input_ids'])
        mhc = {k: torch.tensor(v).to(self.device) for k, v in mhc.items()}
        # print(mhc)
        if self.reweight:
            weight = torch.tensor(weight).to(self.device)
            return peptide, alpha, beta, binder, mhc, weight



        return peptide, alpha, beta, binder, mhc


    def all2allmhc_catpep_collate_function(self, batch):

        if self.reweight:
            (alpha, beta, peptide, binder, mhc, weight) = zip(*batch)
        else:
            (alpha, beta, peptide, binder, mhc) = zip(*batch)

        peptide = self.peptokenizer(list(peptide))
#         print(peptide["input_ids"])
#         print("coucou")
#         print(torch.tensor(peptide["input_ids"]))
        
        peptide = {k: torch.tensor(v).to(self.device) for k, v in peptide.items()}#default_collate(peptide)
        
        beta = self.tokenizer(list(beta),  padding="longest", add_special_tokens=True)
        beta = {k: torch.tensor(v).to(self.device) for k, v in beta.items()}

        alpha = self.tokenizer(list(alpha), padding="longest", add_special_tokens=True)
        alpha = {k: torch.tensor(v).to(self.device) for k, v in alpha.items()}
        
        binder =  default_collate(binder).to(self.device)
        mhc = self.mhctok(list(mhc))#default_collate(self.mhctok(list(mhc))['input_ids'])
        mhc = {k: torch.tensor(v).to(self.device) for k, v in mhc.items()}
        # print(mhc)
        if self.reweight:
            weight = torch.tensor(weight).to(self.device)
            return peptide, alpha, beta, binder, mhc, weight



        return peptide, alpha, beta, binder, mhc



#     def _collate_helper(self, examples, tokenizer):
#         length_of_first = examples[0].size(0)
#         are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
#         if are_tensors_same_length:
#             return torch.stack(examples, dim=0)
#         else:
#             if tokenizer._pad_token is None:
#                 raise ValueError(
#                     "You are attVDJ_trainempting to pad samples but the tokenizer you are using"
#                     f" ({tokenizer.__class__.__name__}) does not have one."
#                 )
#             return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)



# datasetTrain = TCRDataset("dataTrans/trainC.csv", tokenizer, target_binder=1)
# datasetValid = TCRDataset("dataTrans/testC.csv", tokenizer)

# from transformers. import 
# from transformers. import 
# from transformers. import 


# from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertOnlyMLMHead, SequenceClassifierOutput


# from ...configuration_utils import PretrainedConfig
# from ...modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
# from ...modeling_utils import PreTrainedModel
# from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
# from ..auto.configuration_auto import AutoConfig
# from ..auto.modeling_auto import AutoModel, AutoModelForCausalLM
# from .configuration_encoder_decoder import EncoderDecoderConfig


class ClassifCausalLMOutputWithCrossAttentions(ModelOutput):
    lm_loss: Optional[torch.FloatTensor] = None
    lossCLS: Optional[torch.FloatTensor] = None
    pooled_output: Optional[torch.FloatTensor] = None
    clf_logits: torch.FloatTensor = None
    lm_logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None


class BertLastPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor, targetind) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        # print("coucou1",targetind[0], hidden_states[0,targetind[0]])
        ele = torch.arange(0, hidden_states.shape[0])

        first_token_tensor = hidden_states[ele.long(), targetind.long()]#.gather(1, targetind.view(-1,1))#hidden_states[:, -1]
        # print("coucou2", first_token_tensor)
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output



class ED_BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.LMcls = BertOnlyMLMHead(config)
        self.alpha = 0.0
        self.pad_token_id = config.pad_token_id
        print("self.pad_token_id", self.pad_token_id)
        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.LMcls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.LMcls.predictions.decoder = new_embeddings

    # @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # @add_code_sample_docstrings(
    #     processor_class=_TOKENIZER_FOR_DOC,
    #     checkpoint=_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION,
    #     output_type=SequenceClassifierOutput,
    #     config_class=_CONFIG_FOR_DOC,
    #     expected_output=_SEQ_CLASS_EXPECTED_OUTPUT,
    #     expected_loss=_SEQ_CLASS_EXPECTED_LOSS,
    # )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = True# return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
          use_cache = False

        # get clfPosition:
        temp = input_ids != self.pad_token_id
        targetind  = torch.sum(temp, dim=1) - 1

        # print("coucou3", input_ids, targetind)

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.LMcls(sequence_output)
        pooled_output =  self.pooler(sequence_output, targetind) if self.pooler is not None else None

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        labelsCLS = labels[0]
        labelsLM = labels[1]
        lossCLS = None
        if labelsCLS is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labelsCLS.dtype == torch.long or labelsCLS.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    lossCLS = loss_fct(logits.squeeze(), labelsCLS.squeeze())
                else:
                    lossCLS = loss_fct(logits, labelsCLS)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                lossCLS = loss_fct(logits.view(-1, self.num_labels), labelsCLS.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                lossCLS = loss_fct(logits, labelsCLS)

        
        lm_loss = None
        if labelsLM is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
            labelsLM = labelsLM[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss(ignore_index=self.pad_token_id)
            # print(self.pad_token_id)
            lm_loss = loss_fct(shifted_prediction_scores.view(-1, self.config.vocab_size), labelsLM.view(-1))


        # if not return_dict:
        #     output = (logits,) + outputs[2:]
        #     return ((lossCLS,lm_loss,) + output) if loss is not None else output

        return ClassifCausalLMOutputWithCrossAttentions(
            lm_loss=lm_loss,
            lossCLS=lossCLS,
            pooled_output=pooled_output,
            clf_logits  = logits,
            lm_logits=prediction_scores,
            past_key_values= outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions = outputs.cross_attentions
        )





class ED_LMOutput(ModelOutput):
    """
    Base class for sequence-to-sequence language models outputs.
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """

    clf_loss: Optional[torch.FloatTensor] = None
    clf_logits: Optional[torch.FloatTensor] = None
    decoder_outputsA = None
    encoder_outputsA = None
    decoder_outputsB = None
    encoder_outputsB = None
    decoder_outputsE = None
    encoder_outputsE = None
    # logits: torch.FloatTensor = None
    # past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    # decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    # encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "EncoderDecoderConfig"

DEPRECATION_WARNING = (
    "Version v4.12.0 introduces a better way to train encoder-decoder models by computing the loss inside the"
    " encoder-decoder framework rather than in the decoder itself. You may observe training discrepancies if"
    " fine-tuning a model trained with versions anterior to 4.12.0. The decoder_input_ids are now created based on the"
    " labels, no need to pass them yourself anymore."
)

ENCODER_DECODER_START_DOCSTRING = r"""
    This class can be used to initialize a sequence-to-sequence model with any pretrained autoencoding model as the
    encoder and any pretrained autoregressive model as the decoder. The encoder is loaded via
    [`~AutoModel.from_pretrained`] function and the decoder is loaded via [`~AutoModelForCausalLM.from_pretrained`]
    function. Cross-attention layers are automatically added to the decoder and should be fine-tuned on a downstream
    generative task, like summarization.
    The effectiveness of initializing sequence-to-sequence models with pretrained checkpoints for sequence generation
    tasks was shown in [Leveraging Pre-trained Checkpoints for Sequence Generation
    Tasks](https://arxiv.org/abs/1907.12461) by Sascha Rothe, Shashi Narayan, Aliaksei Severyn. Michael Matena, Yanqi
    Zhou, Wei Li, Peter J. Liu.
    After such an Encoder Decoder model has been trained/fine-tuned, it can be saved/loaded just like any other models
    (see the examples for more information).
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)
    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.
    Parameters:
        config ([`EncoderDecoderConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

ENCODER_DECODER_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.
            Indices can be obtained using [`PreTrainedTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary.
            Indices can be obtained using [`PreTrainedTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
            [What are input IDs?](../glossary#input-ids)
            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).
            For training, `decoder_input_ids` are automatically created by the model by shifting the `labels` to the
            right, replacing -100 by the `pad_token_id` and prepending them with the `decoder_start_token_id`.
        decoder_attention_mask (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.
        encoder_outputs (`tuple(torch.FloatTensor)`, *optional*):
            This tuple must consist of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
            `last_hidden_state` (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) is a tensor
            of hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the
            decoder.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        decoder_inputs_embeds (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded
            representation. This is useful if you want more control over how to convert `decoder_input_ids` indices
            into associated vectors than the model's internal embedding lookup matrix.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss for the decoder. Indices should be in `[-100, 0,
            ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            If set to `True`, the model will return a [`~utils.Seq2SeqLMOutput`] instead of a plain tuple.
        kwargs (*optional*): Remaining dictionary of keyword arguments. Keyword arguments come in two flavors:
            - Without a prefix which will be input as `**encoder_kwargs` for the encoder forward function.
            - With a *decoder_* prefix which will be input as `**decoder_kwargs` for the decoder forward function.
"""


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    if decoder_start_token_id is None:
        raise ValueError("Make sure to set the decoder_start_token_id attribute of the model's configuration.")
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("Make sure to set the pad_token_id attribute of the model's configuration.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


# @add_start_docstrings(ENCODER_DECODER_START_DOCSTRING)
class ED_MultiTransformerModel(PreTrainedModel):
    r"""
    [`EncoderDecoderModel`] is a generic model class that will be instantiated as a transformer architecture with one
    of the base model classes of the library as encoder and another one as decoder when created with the
    :meth*~transformers.AutoModel.from_pretrained* class method for the encoder and
    :meth*~transformers.AutoModelForCausalLM.from_pretrained* class method for the decoder.
    """
    config_class = EncoderDecoderConfig
    #config_class = multiTransConfig
    base_model_prefix = "encoder_decoder"

    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        encoderA: Optional[PreTrainedModel] = None,
        decoderA: Optional[PreTrainedModel] = None,
        encoderB: Optional[PreTrainedModel] = None,
        decoderB: Optional[PreTrainedModel] = None,
        encoderE: Optional[PreTrainedModel] = None,
        decoderE: Optional[PreTrainedModel] = None,
    ):
        if config is None and (encoderA is None or decoderA is None):
            raise ValueError("Either a configuration or an encoder and a decoder has to be provided.")
        if config is None:
            config = EncoderDecoderConfig.from_encoder_decoder_configs(encoderA.config, decoderA.config)
            #config = EncoderDecoderConfig.from_encoder_decoder_configs(encoderA.config, decoderA.config,encoderB.config, decoderB.config,encoderE.config, decoderE.config)
        else:
            if not isinstance(config, self.config_class):
                raise ValueError(f"Config: {config} has to be of type {self.config_class}")

        if config.decoder.cross_attention_hidden_size is not None:
            if config.decoder.cross_attention_hidden_size != config.encoder.hidden_size:
                raise ValueError(
                    "If `cross_attention_hidden_size` is specified in the decoder's configuration, it has to be equal"
                    f" to the encoder's `hidden_size`. Got {config.decoder.cross_attention_hidden_size} for"
                    f" `config.decoder.cross_attention_hidden_size` and {config.encoder.hidden_size} for"
                    " `config.encoder.hidden_size`."
                )

        # initialize with config
        super().__init__(config)

        if encoderA is None:
            from ..auto.modeling_auto import AutoModel

            encoderA = AutoModel.from_config(config.encoder)
        if encoderE is None:
            from ..auto.modeling_auto import AutoModel

            encoderE = AutoModel.from_config(config.encoder)
        if encoderB is None:
            from ..auto.modeling_auto import AutoModel

            encoderB = AutoModel.from_config(config.encoder)

        if decoderA is None:
            from ..auto.modeling_auto import AutoModelForCausalLM
            decoderA = AutoModelForCausalLM.from_config(config.decoder)
        if decoderB is None:
            from ..auto.modeling_auto import AutoModelForCausalLM
            decoderB = AutoModelForCausalLM.from_config(config.decoder)
        if decoderE is None:
            from ..auto.modeling_auto import AutoModelForCausalLM
            decoderE = AutoModelForCausalLM.from_config(config.decoder)

        self.encoderA = encoderA
        self.decoderA = decoderA
        self.encoderB = encoderB
        self.decoderB = decoderB
        self.encoderE = encoderE
        self.decoderE = decoderE
        self.num_labels = 2
        self.MLMHeadA =  BertOnlyMLMHead(decoderA.config)
        self.MLMHeadB =  BertOnlyMLMHead(decoderB.config)
        self.MLMHeadE =  BertOnlyMLMHead(decoderE.config)
        self.classifier = nn.Linear(3*decoderA.config.hidden_size, 2)
        self.mhc_embeddings = nn.Embedding(encoderA.config.mhc_vocab_size, encoderA.config.hidden_size)
        if self.encoderA.config.to_dict() != self.config.encoder.to_dict():
            logger.warning(
                f"Config of the encoder: {self.encoderA.__class__} is overwritten by shared encoder config:"
                f" {self.config.encoder}"
            )
        if self.decoderA.config.to_dict() != self.config.decoder.to_dict():
            logger.warning(
                f"Config of the decoder: {self.decoderA.__class__} is overwritten by shared decoder config:"
                f" {self.config.decoder}"
            )

        # make sure that the individual model's config refers to the shared config
        # so that the updates to the config will be synced
        # self.encoderA.config = self.config.encoder
        # self.decoderA.config = self.config.decoder
        # self.encoderB.config = self.config.encoder
        # self.decoderB.config = self.config.decoder
        # self.encoderE.config = self.config.encoder
        # self.decoderE.config = self.config.decoder
        # encoder outputs might need to be projected to different dimension for decoder
        if (
            self.encoderA.config.hidden_size != self.decoderA.config.hidden_size
            and self.decoderA.config.cross_attention_hidden_size is None
        ):
            self.enc_to_dec_proj = nn.Linear(self.encoderA.config.hidden_size, self.decoderA.config.hidden_size)

        if self.encoderA.get_output_embeddings() is not None:
            raise ValueError(
                f"The encoder {self.encoderA} should not have a LM Head. Please use a model without LM Head"
            )

        # tie encoder, decoder weights if config set accordingly
        self.tie_weights()

    def tie_weights(self):
        # tie encoder & decoder if needed
        if self.config.tie_encoder_decoder:
            # tie encoder and decoder base model
            decoder_base_model_prefix = self.decoder.base_model_prefix
            self._tie_encoder_decoder_weights(
                self.encoder, self.decoder._modules[decoder_base_model_prefix], self.decoder.base_model_prefix
            )

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_input_embeddings(self):
        return self.encoder.get_input_embeddings()

    def get_output_embeddings(self):
        return self.decoder.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        return self.decoder.set_output_embeddings(new_embeddings)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # At the moment fast initialization is not supported for composite models
        if kwargs.get("_fast_init", False):
            logger.warning(
                "Fast initialization is currently not supported for EncoderDecoderModel. "
                "Falling back to slow initialization..."
            )
        kwargs["_fast_init"] = False
        return super().from_pretrained(*args, **kwargs)

    @classmethod
    def from_encoder_decoder_pretrained(
        cls,
        encoder_pretrained_model_name_or_path: str = None,
        decoder_pretrained_model_name_or_path: str = None,
        *model_args,
        **kwargs
    ) -> PreTrainedModel:
        r"""
        Instantiate an encoder and a decoder from one or two base classes of the library from pretrained model
        checkpoints.
        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated). To train
        the model, you need to first set it back in training mode with `model.train()`.
        Params:
            encoder_pretrained_model_name_or_path (`str`, *optional*):
                Information necessary to initiate the encoder. Can be either:
                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
                      user or organization name, like `dbmdz/bert-base-german-cased`.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                    - A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
                      this case, `from_tf` should be set to `True` and a configuration object should be provided as
                      `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
                      PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
            decoder_pretrained_model_name_or_path (`str`, *optional*, defaults to `None`):
                Information necessary to initiate the decoder. Can be either:
                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
                      user or organization name, like `dbmdz/bert-base-german-cased`.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                    - A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
                      this case, `from_tf` should be set to `True` and a configuration object should be provided as
                      `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
                      PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
            model_args (remaining positional arguments, *optional*):
                All remaining positional arguments will be passed to the underlying model's `__init__` method.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
                `output_attentions=True`).
                - To update the encoder configuration, use the prefix *encoder_* for each configuration parameter.
                - To update the decoder configuration, use the prefix *decoder_* for each configuration parameter.
                - To update the parent model configuration, do not use a prefix for each configuration parameter.
                Behaves differently depending on whether a `config` is provided or automatically loaded.
        Example:
        ```python
        >>> from transformers import EncoderDecoderModel
        >>> # initialize a bert2bert from two pretrained BERT models. Note that the cross-attention layers will be randomly initialized
        >>> model = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased", "bert-base-uncased")
        >>> # saving model after fine-tuning
        >>> model.save_pretrained("./bert2bert")
        >>> # load fine-tuned model
        >>> model = EncoderDecoderModel.from_pretrained("./bert2bert")
        ```"""

        kwargs_encoder = {
            argument[len("encoder_") :]: value for argument, value in kwargs.items() if argument.startswith("encoder_")
        }

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        # remove encoder, decoder kwargs from kwargs
        for key in kwargs_encoder.keys():
            del kwargs["encoder_" + key]
        for key in kwargs_decoder.keys():
            del kwargs["decoder_" + key]

        # Load and initialize the encoder and decoder
        # The distinction between encoder and decoder at the model level is made
        # by the value of the flag `is_decoder` that we need to set correctly.
        encoder = kwargs_encoder.pop("model", None)
        if encoder is None:
            if encoder_pretrained_model_name_or_path is None:
                raise ValueError(
                    "If `encoder_model` is not defined as an argument, a `encoder_pretrained_model_name_or_path` has "
                    "to be defined."
                )

            if "config" not in kwargs_encoder:
                encoder_config, kwargs_encoder = AutoConfig.from_pretrained(
                    encoder_pretrained_model_name_or_path, **kwargs_encoder, return_unused_kwargs=True
                )

                if encoder_config.is_decoder is True or encoder_config.add_cross_attention is True:
                    logger.info(
                        f"Initializing {encoder_pretrained_model_name_or_path} as a encoder model "
                        "from a decoder model. Cross-attention and casual mask are disabled."
                    )
                    encoder_config.is_decoder = False
                    encoder_config.add_cross_attention = False

                kwargs_encoder["config"] = encoder_config

            encoder = AutoModel.from_pretrained(encoder_pretrained_model_name_or_path, *model_args, **kwargs_encoder)

        decoder = kwargs_decoder.pop("model", None)
        if decoder is None:
            if decoder_pretrained_model_name_or_path is None:
                raise ValueError(
                    "If `decoder_model` is not defined as an argument, a `decoder_pretrained_model_name_or_path` has "
                    "to be defined."
                )

            if "config" not in kwargs_decoder:
                decoder_config, kwargs_decoder = AutoConfig.from_pretrained(
                    decoder_pretrained_model_name_or_path, **kwargs_decoder, return_unused_kwargs=True
                )

                if decoder_config.is_decoder is False or decoder_config.add_cross_attention is False:
                    logger.info(
                        f"Initializing {decoder_pretrained_model_name_or_path} as a decoder model. Cross attention"
                        f" layers are added to {decoder_pretrained_model_name_or_path} and randomly initialized if"
                        f" {decoder_pretrained_model_name_or_path}'s architecture allows for cross attention layers."
                    )
                    decoder_config.is_decoder = True
                    decoder_config.add_cross_attention = True

                kwargs_decoder["config"] = decoder_config

            if kwargs_decoder["config"].is_decoder is False or kwargs_decoder["config"].add_cross_attention is False:
                logger.warning(
                    f"Decoder model {decoder_pretrained_model_name_or_path} is not initialized as a decoder. "
                    f"In order to initialize {decoder_pretrained_model_name_or_path} as a decoder, "
                    "make sure that the attributes `is_decoder` and `add_cross_attention` of `decoder_config` "
                    "passed to `.from_encoder_decoder_pretrained(...)` are set to `True` or do not pass a "
                    "`decoder_config` to `.from_encoder_decoder_pretrained(...)`"
                )

            decoder = AutoModelForCausalLM.from_pretrained(decoder_pretrained_model_name_or_path, **kwargs_decoder)

        # instantiate config with corresponding kwargs
        config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config, **kwargs)
        return cls(encoder=encoder, decoder=decoder, config=config)

    # @add_start_docstrings_to_model_forward(ENCODER_DECODER_INPUTS_DOCSTRING) 
    # @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = (None, None,None),
        attention_mask: Optional[torch.FloatTensor] =  (None, None,None),
        # decoder_input_ids: Optional[torch.LongTensor] =  (None, None,None),
        # decoder_attention_mask: Optional[torch.BoolTensor] =  (None, None,None),
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = (None, None,None),
        past_key_values: Tuple[Tuple[torch.FloatTensor]] =  (None, None,None),
        inputs_embeds: Optional[torch.FloatTensor] = (None, None,None),
        # decoder_inputs_embeds: Optional[torch.FloatTensor] =  (None, None,None),
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] =  (None, None,None),
        output_attentions: Optional[bool] =  None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = (None, None,None),
        mhc=None,
        **kwargs,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        input_idsA=input_ids[0]
        input_idsB=input_ids[1]
        input_idsE=input_ids[2]
        
        attention_maskA=attention_mask[0]
        attention_maskB=attention_mask[1]
        attention_maskE=attention_mask[2]

        encoder_outputsA=encoder_outputs[0]
        encoder_outputsB=encoder_outputs[1]
        encoder_outputsE=encoder_outputs[2]

        past_key_valuesA=past_key_values[0]
        past_key_valuesB=past_key_values[1]
        past_key_valuesE=past_key_values[2]

        inputs_embedsA=inputs_embeds[0]
        inputs_embedsB=inputs_embeds[1]
        inputs_embedsE=inputs_embeds[2]
        r"""
        Returns:
        Examples:
        ```python
        >>> from transformers import EncoderDecoderModel, BertTokenizer
        >>> import torch
        >>> tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        >>> model = EncoderDecoderModel.from_encoder_decoder_pretrained(
        ...     "bert-base-uncased", "bert-base-uncased"
        ... )  # initialize Bert2Bert from pre-trained checkpoints
        >>> # training
        >>> model.config.decoder_start_token_id = tokenizer.cls_token_id
        >>> model.config.pad_token_id = tokenizer.pad_token_id
        >>> model.config.vocab_size = model.config.decoder.vocab_size
        >>> input_ids = tokenizer("This is a really long text", return_tensors="pt").input_ids
        >>> labels = tokenizer("This is the corresponding summary", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=input_ids)
        >>> loss, logits = outputs.loss, outputs.logits
        >>> # save and load from pretrained
        >>> model.save_pretrained("bert2bert")
        >>> model = EncoderDecoderModel.from_pretrained("bert2bert")
        >>> # generation
        >>> generated = model.generate(input_ids)
        ```"""






        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        if encoder_outputsA is None:
            encoder_outputsA = self.encoderA(
                input_ids=input_idsA,
                attention_mask=attention_maskA,
                inputs_embeds=inputs_embedsA,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
            )
        elif isinstance(encoder_outputsA, tuple):
            encoder_outputsA = BaseModelOutput(*encoder_outputsA)


        if encoder_outputsB is None:
            encoder_outputsB = self.encoderB(
                input_ids=input_idsB,
                attention_mask=attention_maskB,
                inputs_embeds=inputs_embedsB,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
            )
        elif isinstance(encoder_outputsB, tuple):
            encoder_outputsB = BaseModelOutput(*encoder_outputsB)

                
        if encoder_outputsE is None:
            encoder_outputsE = self.encoderE(
                input_ids=input_idsE,
                attention_mask=attention_maskE,
                inputs_embeds=inputs_embedsE,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
            )
        elif isinstance(encoder_outputsE, tuple):
            encoder_outputsE = BaseModelOutput(*encoder_outputsE)

        encoder_hidden_statesA = encoder_outputsA[0]
        encoder_hidden_statesB = encoder_outputsB[0]
        encoder_hidden_statesE = encoder_outputsE[0]
        # optionally project encoder_hidden_states
        if (
            self.encoderA.config.hidden_size != self.decoderA.config.hidden_size
            and self.decoderA.config.cross_attention_hidden_size is None
        ):
            encoder_hidden_statesA = self.enc_to_dec_proj(encoder_hidden_statesA)

        if (
            self.encoderB.config.hidden_size != self.decoderB.config.hidden_size
            and self.decoderB.config.cross_attention_hidden_size is None
        ):
            encoder_hidden_statesB = self.enc_to_dec_proj(encoder_hidden_statesB)

        if (
            self.encoderE.config.hidden_size != self.decoderE.config.hidden_size
            and self.decoderE.config.cross_attention_hidden_size is None
        ):
            encoder_hidden_statesE = self.enc_to_dec_proj(encoder_hidden_statesE)

        # if (labels is not None) and (decoder_input_ids is None and decoder_inputs_embeds is None):
        #     decoder_input_ids = shift_tokens_right(
        #         labels, self.config.pad_token_id, self.config.decoder_start_token_id
        #     )
        # print(dict(mhc))
        # print(type(mhc["input_ids"]))
        #mhc = default_collate(dict(mhc))
        # print(mhc["input_ids"])
        # print(torch.tensor(mhc["input_ids"]))
        mhc_encoded = self.mhc_embeddings(mhc["input_ids"])
        mhc_attention_mask = mhc["attention_mask"]
        # Decode
        labelsA = (labels, input_idsA)
        decoder_outputsA = self.decoderA(
            input_ids = input_idsA,
            attention_mask = attention_maskA,
            encoder_hidden_states = torch.cat([mhc_encoded,encoder_hidden_statesB, encoder_hidden_statesE], dim=1),
            encoder_attention_mask = torch.cat([mhc_attention_mask, attention_maskB, attention_maskE], dim=1),
            inputs_embeds = inputs_embedsA,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            labels=labelsA,
            use_cache=use_cache,
            past_key_values=past_key_valuesA,
            return_dict=return_dict,
            **kwargs_decoder,
        )
        pooled_outputA = decoder_outputsA.pooled_output

        labelsB = (labels, input_idsB)
        decoder_outputsB = self.decoderB(
            input_ids = input_idsB,
            attention_mask = attention_maskB,
            encoder_hidden_states = torch.cat([mhc_encoded,encoder_hidden_statesA, encoder_hidden_statesE], dim=1),
            encoder_attention_mask = torch.cat([mhc_attention_mask,attention_maskA, attention_maskE], dim=1),
            inputs_embeds = inputs_embedsB,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            labels=labelsB,
            use_cache=use_cache,
            past_key_values=past_key_valuesB,
            return_dict=return_dict,
            **kwargs_decoder,
        )
        pooled_outputB = decoder_outputsB.pooled_output

        labelsE = (labels, input_idsE)
        decoder_outputsE = self.decoderE(
            input_ids = input_idsE,
            attention_mask = attention_maskE,
            encoder_hidden_states = torch.cat([mhc_encoded,encoder_hidden_statesA, encoder_hidden_statesB], dim=1),
            encoder_attention_mask = torch.cat([mhc_attention_mask,attention_maskA, attention_maskB], dim=1),
            inputs_embeds = inputs_embedsE,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            labels=labelsE,
            use_cache=use_cache,
            past_key_values=past_key_valuesE,
            return_dict=return_dict,
            **kwargs_decoder,
        )
        pooled_outputE= decoder_outputsE.pooled_output

        pooled_output = torch.cat([pooled_outputA,pooled_outputB,pooled_outputE], dim=1)
        logits = self.classifier(pooled_output)
        labelsCLS = labels
        lossCLS = None
        if labelsCLS is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labelsCLS.dtype == torch.long or labelsCLS.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    lossCLS = loss_fct(logits.squeeze(), labelsCLS.squeeze())
                else:
                    lossCLS = loss_fct(logits, labelsCLS)
            elif self.config.problem_type == "single_label_classification":
                if self.reweight == True:
                    loss_fct = CrossEntropyLoss(reduction="none")
                    lossCLS = loss_fct(logits.view(-1, self.num_labels), labelsCLS.view(-1))
                else:
                    loss_fct = CrossEntropyLoss()
                    lossCLS = loss_fct(logits.view(-1, self.num_labels), labelsCLS.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                lossCLS = loss_fct(logits, labelsCLS)




        # Compute loss independent from decoder (as some shift the logits inside them)
        loss = None
        # if labels is not None:
        #     warnings.warn(DEPRECATION_WARNING, FutureWarning)
        #     logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
        #     loss_fct = CrossEntropyLoss()
        #     loss = loss_fct(logits.reshape(-1, self.decoder.config.vocab_size), labels.view(-1))

        if not return_dict:
            if loss is not None:
                return (loss,) + decoder_outputs + encoder_outputs
            else:
                return decoder_outputs + encoder_outputs

        return ED_LMOutput(
            loss = lossCLS,
            clf_logits=logits,
            encoder_outputsA = encoder_outputsA,
            decoder_outputsA = decoder_outputsA,
            encoder_outputsB = encoder_outputsB,
            decoder_outputsB = decoder_outputsB,
            encoder_outputsE = encoder_outputsE,
            decoder_outputsE = decoder_outputsE,
        )
        
        # Seq2SeqLMOutput(
        #     loss=loss,
        #     logits=decoder_outputs.logits,
        #     past_key_values=decoder_outputs.past_key_values,
        #     decoder_hidden_states=decoder_outputs.hidden_states,
        #     decoder_attentions=decoder_outputs.attentions,
        #     cross_attentions=decoder_outputs.cross_attentions,
        #     encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        #     encoder_hidden_states=encoder_outputs.hidden_states,
        #     encoder_attentions=encoder_outputs.attentions,
        # )

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    def prepare_inputs_for_generation(
        self, input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    ):
        decoder_inputs = self.decoder.prepare_inputs_for_generation(input_ids, past=past)
        decoder_attention_mask = decoder_inputs["attention_mask"] if "attention_mask" in decoder_inputs else None
        input_dict = {
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_input_ids": decoder_inputs["input_ids"],
            "encoder_outputs": encoder_outputs,
            "past_key_values": decoder_inputs["past_key_values"],
            "use_cache": use_cache,
        }
        return input_dict

    def resize_token_embeddings(self, *args, **kwargs):
        raise NotImplementedError(
            "Resizing the embedding layers via the EncoderDecoderModel directly is not supported. Please use the"
            " respective methods of the wrapped objects (model.encoder.resize_token_embeddings(...) or"
            " model.decoder.resize_token_embeddings(...))"
        )

    def _reorder_cache(self, past, beam_idx):
        # apply decoder cache reordering here
        return self.decoder._reorder_cache(past, beam_idx)


    def set_reweight(self):
        self.reweight = True






































# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print("Using device:", device)



def compute_loss(predictions, targets, criterion):
    """Compute our custom loss"""
    if len(targets)>0:
      predictions = predictions[:, :-1, :].contiguous()
      targets = targets[:, 1:]

      rearranged_output = predictions.view(predictions.shape[0]*predictions.shape[1], -1)
      rearranged_target = targets.contiguous().view(-1)

      loss = criterion(rearranged_output, rearranged_target)
    else:
      loss = 0

    return loss

def compute_loss_rw(predictions, targets, w, criterion):
    """Compute our custom loss"""
    if len(targets)>0:
      predictions = predictions[:, :-1, :].contiguous()
      targets = targets[:, 1:]

      rearranged_output = predictions.view(predictions.shape[0]*predictions.shape[1], -1)
      rearranged_target = targets.contiguous().view(-1)

      losses = criterion(rearranged_output, rearranged_target).view(predictions.shape[0],predictions.shape[1])
      losses = losses.sum(axis=1)
      loss = torch.sum(torch.mul(losses, w))

    else:
      loss = 0

    return loss


def MLM_Loss(encoder, head, masker, inputsid, inputsam, observedmask):
    if sum(observedmask) !=0:
        input_ids, labels = masker.torch_mask_tokens(inputsid[observedmask])
        attention_mask = inputsam
        outputs = encoder(
            input_ids,
            attention_mask=attention_mask[observedmask],
            return_dict=True,
        )

        sequence_output = outputs.last_hidden_state
        prediction_scores = head(sequence_output)

        masked_lm_loss = None
        loss_fct = CrossEntropyLoss(reduction='sum')  # -100 index = padding token
        masked_lm_loss = loss_fct(prediction_scores.view(-1, encoder.config.vocab_size), labels.view(-1))
        return masked_lm_loss
    else:
        return 0




def MLM_Loss_rw(encoder, head, masker, inputsid, inputsam, observedmask, w):
    if sum(observedmask) !=0:
        input_ids, labels = masker.torch_mask_tokens(inputsid[observedmask])
        w = w[observedmask]
        attention_mask = inputsam
        outputs = encoder(
            input_ids,
            attention_mask=attention_mask[observedmask],
            return_dict=True,
        )

        sequence_output = outputs.last_hidden_state
        prediction_scores = head(sequence_output)

        masked_lm_loss = None
        loss_fct = CrossEntropyLoss(reduction='none')  # -100 index = padding token

        masked_lm_loss = loss_fct(prediction_scores.view(-1, encoder.config.vocab_size), labels.view(-1))
        masked_lm_loss = masked_lm_loss.view()
        return masked_lm_loss
    else:
        return 0


def train_unsupervised(model, optimizer, masker, train_dataloader, criterion, alph = 1.0):
    model.train()
    # epoch_loss = 0
    # epoch_clfloss=0
    epoch_lm_lossA = 0
    epoch_lm_lossB = 0
    epoch_lm_lossE = 0
    epoch_mlm_lossA = 0
    epoch_mlm_lossB = 0
    epoch_mlm_lossE = 0
    count_A = 0
    count_B = 0
    count_E = 0
    for i, (peptide, alpha, beta, binder, mhc) in enumerate(train_dataloader):
        optimizer.zero_grad()
        peptide_input = peptide['input_ids']
        peptide_mask= peptide["attention_mask"]
        peptide_tokentype = peptide['token_type_ids']
        alpha_input = alpha['input_ids']
        alpha_mask = alpha["attention_mask"]
        alpha_tokentype = alpha['token_type_ids']
        beta_input = beta['input_ids']
        beta_mask = beta["attention_mask"]
        beta_tokentype = beta['token_type_ids']
        
        # binder = binder
        # positive_mask = binder == 1
        alpha_observed_mask = alpha_input.clone().detach()[:,1] != 4
        beta_observed_mask = beta_input.clone().detach()[:,1] != 4
        peptide_observed_mask = peptide_input.clone().detach()[:,1] != 4
        # lm_labels = alphabeta_output.clone()
        clf_label = binder.clone()
        labels = clf_label

        out = model(input_ids=(alpha_input,beta_input,peptide_input), attention_mask=(alpha_mask,beta_mask,peptide_mask),
                                        labels=labels, mhc=mhc)

        loss_clf = out.loss #decoder_outputs.lossCLS
        lm_lossA = out.decoder_outputsA.lm_loss
        lm_lossB = out.decoder_outputsB.lm_loss
        lm_lossE = out.decoder_outputsE.lm_loss
        # clf_lossA = out.decoder_outputsA.lossCLS
        # clf_lossB = out.decoder_outputsB.lossCLS
        # clf_lossE = out.decoder_outputsE.lossCLS
        prediction_scoresA = out.decoder_outputsA.lm_logits
        predictionsA = F.log_softmax(prediction_scoresA, dim=2)
        prediction_scoresB = out.decoder_outputsB.lm_logits
        predictionsB = F.log_softmax(prediction_scoresB, dim=2)
        prediction_scoresE = out.decoder_outputsE.lm_logits
        predictionsE = F.log_softmax(prediction_scoresE, dim=2)
        lossa = compute_loss(predictionsA[alpha_observed_mask], alpha_input[alpha_observed_mask], criterion) 
        lossb = compute_loss(predictionsB[beta_observed_mask], beta_input[beta_observed_mask],criterion) 
        losse = compute_loss(predictionsE[peptide_observed_mask], peptide_input[peptide_observed_mask], criterion)
        # print("lossa", lossa, lm_lossA)
        # print("lossb", lossb, lm_lossB)
        # print("losse", losse, lm_lossE)
        mlm_lossA = MLM_Loss(model.encoderA, model.MLMHeadA, masker, alpha_input, alpha_mask, alpha_observed_mask)
        mlm_lossB = MLM_Loss(model.encoderB, model.MLMHeadB, masker, beta_input, beta_mask, beta_observed_mask)
        mlm_lossE = MLM_Loss(model.encoderE, model.MLMHeadE, masker, peptide_input, peptide_mask, peptide_observed_mask)


        # if loss !=0:
        #   epoch_loss += loss.item()
        # loss = alph*loss + (1-alph)*loss_clf
        # if loss ==0 and alph ==1:
        #     epoch_clfloss += loss_clf.item()
        #     continue
        
        loss = mlm_lossA+mlm_lossB+mlm_lossE+alph*(lossa+lossb+losse)
        # loss = lossa+lossb+losse

        loss.backward()
        # loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # print(loss_clf)#model.decoder.pooler.dense.weight)
        optimizer.step()
        # print(model.decoder.bert.pooler.dense.weight)
        count_A += sum(alpha_observed_mask)
        count_B += sum(beta_observed_mask)
        count_E += sum(peptide_observed_mask)
        epoch_lm_lossA += lossa
        epoch_lm_lossB += lossb
        epoch_lm_lossE += losse
        epoch_mlm_lossA += mlm_lossA
        epoch_mlm_lossB += mlm_lossB
        epoch_mlm_lossE += mlm_lossE

    epoch_lm_lossA /= count_A
    epoch_lm_lossB /= count_B
    epoch_lm_lossE /= count_E
    epoch_mlm_lossA /= count_A
    epoch_mlm_lossB /= count_B
    epoch_mlm_lossE /= count_E

    return epoch_lm_lossA, epoch_lm_lossB, epoch_lm_lossE, epoch_mlm_lossA, epoch_mlm_lossB, epoch_mlm_lossE




def train_unsupervised_reweight(model, optimizer, masker, train_dataloader, criterion, alph = 1.0):
    model.train()
    # epoch_loss = 0
    # epoch_clfloss=0
    epoch_lm_lossA = 0
    epoch_lm_lossB = 0
    epoch_lm_lossE = 0
    epoch_mlm_lossA = 0
    epoch_mlm_lossB = 0
    epoch_mlm_lossE = 0
    count_A = 0
    count_B = 0
    count_E = 0
    for i, (peptide, alpha, beta, binder, mhc, w) in enumerate(train_dataloader):
        optimizer.zero_grad()
        peptide_input = peptide['input_ids']
        peptide_mask= peptide["attention_mask"]
        peptide_tokentype = peptide['token_type_ids']
        alpha_input = alpha['input_ids']
        alpha_mask = alpha["attention_mask"]
        alpha_tokentype = alpha['token_type_ids']
        beta_input = beta['input_ids']
        beta_mask = beta["attention_mask"]
        beta_tokentype = beta['token_type_ids']
        
        # binder = binder
        # positive_mask = binder == 1
        alpha_observed_mask = alpha_input.clone().detach()[:,1] != 4
        beta_observed_mask = beta_input.clone().detach()[:,1] != 4
        peptide_observed_mask = peptide_input.clone().detach()[:,1] != 4
        # lm_labels = alphabeta_output.clone()
        clf_label = binder.clone()
        labels = clf_label

        out = model(input_ids=(alpha_input,beta_input,peptide_input), attention_mask=(alpha_mask,beta_mask,peptide_mask),
                                        labels=labels, mhc=mhc)

        loss_clf = out.loss #decoder_outputs.lossCLS
        lm_lossA = out.decoder_outputsA.lm_loss
        lm_lossB = out.decoder_outputsB.lm_loss
        lm_lossE = out.decoder_outputsE.lm_loss
        # clf_lossA = out.decoder_outputsA.lossCLS
        # clf_lossB = out.decoder_outputsB.lossCLS
        # clf_lossE = out.decoder_outputsE.lossCLS
        prediction_scoresA = out.decoder_outputsA.lm_logits
        predictionsA = F.log_softmax(prediction_scoresA, dim=2)
        prediction_scoresB = out.decoder_outputsB.lm_logits
        predictionsB = F.log_softmax(prediction_scoresB, dim=2)
        prediction_scoresE = out.decoder_outputsE.lm_logits
        predictionsE = F.log_softmax(prediction_scoresE, dim=2)
        lossa = compute_loss(predictionsA[alpha_observed_mask], alpha_input[alpha_observed_mask], criterion) 
        lossb = compute_loss(predictionsB[beta_observed_mask], beta_input[beta_observed_mask],criterion) 
        losse = compute_loss(predictionsE[peptide_observed_mask], peptide_input[peptide_observed_mask], criterion)
        # print("lossa", lossa, lm_lossA)
        # print("lossb", lossb, lm_lossB)
        # print("losse", losse, lm_lossE)
        mlm_lossA = MLM_Loss(model.encoderA, model.MLMHeadA, masker, alpha_input, alpha_mask, alpha_observed_mask)
        mlm_lossB = MLM_Loss(model.encoderB, model.MLMHeadB, masker, beta_input, beta_mask, beta_observed_mask)
        mlm_lossE = MLM_Loss(model.encoderE, model.MLMHeadE, masker, peptide_input, peptide_mask, peptide_observed_mask)


        # if loss !=0:
        #   epoch_loss += loss.item()
        # loss = alph*loss + (1-alph)*loss_clf
        # if loss ==0 and alph ==1:
        #     epoch_clfloss += loss_clf.item()
        #     continue
        
        loss = mlm_lossA+mlm_lossB+mlm_lossE+alph*(lossa+lossb+losse)
        # loss = lossa+lossb+losse

        loss.backward()
        # loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # print(loss_clf)#model.decoder.pooler.dense.weight)
        optimizer.step()
        # print(model.decoder.bert.pooler.dense.weight)
        count_A += sum(alpha_observed_mask)
        count_B += sum(beta_observed_mask)
        count_E += sum(peptide_observed_mask)
        epoch_lm_lossA += lossa
        epoch_lm_lossB += lossb
        epoch_lm_lossE += losse
        epoch_mlm_lossA += mlm_lossA
        epoch_mlm_lossB += mlm_lossB
        epoch_mlm_lossE += mlm_lossE

    epoch_lm_lossA /= count_A
    epoch_lm_lossB /= count_B
    epoch_lm_lossE /= count_E
    epoch_mlm_lossA /= count_A
    epoch_mlm_lossB /= count_B
    epoch_mlm_lossE /= count_E

    return epoch_lm_lossA, epoch_lm_lossB, epoch_lm_lossE, epoch_mlm_lossA, epoch_mlm_lossB, epoch_mlm_lossE





def train_supervised(model, optimizer, masker, train_dataloader,criterion, alph=1.0):
    model.train()

    epoch_clfloss=0
    epoch_lm_lossA = 0
    epoch_lm_lossB = 0
    epoch_lm_lossE = 0
    epoch_mlm_lossA = 0
    epoch_mlm_lossB = 0
    epoch_mlm_lossE = 0
    count_A = 0
    count_B = 0
    count_E = 0
    for i, (peptide, alpha, beta, binder, mhc) in enumerate(train_dataloader):
        optimizer.zero_grad()
        peptide_input = peptide['input_ids']
        peptide_mask= peptide["attention_mask"]
        peptide_tokentype = peptide['token_type_ids']
        alpha_input = alpha['input_ids']
        alpha_mask = alpha["attention_mask"]
        alpha_tokentype = alpha['token_type_ids']
        beta_input = beta['input_ids']
        beta_mask = beta["attention_mask"]
        beta_tokentype = beta['token_type_ids']
        
        binder = binder
        positive_mask = binder == 1
        alpha_observed_mask = alpha_input.clone().detach()[:,1] != 4
        beta_observed_mask = beta_input.clone().detach()[:,1] != 4
        peptide_observed_mask = peptide_input.clone().detach()[:,1] != 4
        # lm_labels = alphabeta_output.clone()
        clf_label = binder.clone()
        labels = clf_label

        out = model(input_ids=(alpha_input,beta_input,peptide_input), attention_mask=(alpha_mask,beta_mask,peptide_mask),
                                        labels=labels, mhc=mhc)

        loss_clf = out.loss #decoder_outputs.lossCLS
        lm_lossA = out.decoder_outputsA.lm_loss
        lm_lossB = out.decoder_outputsB.lm_loss
        lm_lossE = out.decoder_outputsE.lm_loss
        # clf_lossA = out.decoder_outputsA.lossCLS
        # clf_lossB = out.decoder_outputsB.lossCLS
        # clf_lossE = out.decoder_outputsE.lossCLS
        prediction_scoresA = out.decoder_outputsA.lm_logits
        predictionsA = F.log_softmax(prediction_scoresA, dim=2)
        prediction_scoresB = out.decoder_outputsB.lm_logits
        predictionsB = F.log_softmax(prediction_scoresB, dim=2)
        prediction_scoresE = out.decoder_outputsE.lm_logits
        predictionsE = F.log_softmax(prediction_scoresE, dim=2)
        lossa = compute_loss(predictionsA[alpha_observed_mask&positive_mask], alpha_input[alpha_observed_mask&positive_mask], criterion) 
        lossb = compute_loss(predictionsB[beta_observed_mask&positive_mask], beta_input[beta_observed_mask&positive_mask], criterion) 
        losse = compute_loss(predictionsE[peptide_observed_mask&positive_mask], peptide_input[peptide_observed_mask&positive_mask], criterion)
        # print(lossa, lossb, losse)
        mlm_lossA = MLM_Loss(model.encoderA, model.MLMHeadA, masker, alpha_input, alpha_mask, alpha_observed_mask)
        mlm_lossB = MLM_Loss(model.encoderB, model.MLMHeadB, masker, beta_input, beta_mask, beta_observed_mask)
        mlm_lossE = MLM_Loss(model.encoderE, model.MLMHeadE, masker, peptide_input, peptide_mask, peptide_observed_mask)

        loss = mlm_lossA+mlm_lossB+mlm_lossE+lossa+lossb+losse+alph*loss_clf
        # loss = lossa+lossb+losse+alph*loss_clf
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        count_A += sum(alpha_observed_mask&positive_mask)
        count_B += sum(beta_observed_mask&positive_mask)
        count_E += sum(peptide_observed_mask&positive_mask)
        epoch_lm_lossA += lossa
        epoch_lm_lossB += lossb
        epoch_lm_lossE += losse
        # epoch_mlm_lossA += mlm_lossA
        # epoch_mlm_lossB += mlm_lossB
        # epoch_mlm_lossE += mlm_lossE
        epoch_clfloss += loss_clf

    epoch_lm_lossA /= count_A
    epoch_lm_lossB /= count_B
    epoch_lm_lossE /= count_E
    epoch_mlm_lossA /= count_A
    epoch_mlm_lossB /= count_B
    epoch_mlm_lossE /= count_E
    epoch_clfloss/=len(train_dataloader.dataset)

    return epoch_lm_lossA, epoch_lm_lossB, epoch_lm_lossE, epoch_mlm_lossA, epoch_mlm_lossB, epoch_mlm_lossE, epoch_clfloss






def train_supervised_reweight(model, optimizer, masker, train_dataloader,criterion, alph=1.0):
    model.train()

    epoch_clfloss = 0
    epoch_lm_lossA = 0
    epoch_lm_lossB = 0
    epoch_lm_lossE = 0
    epoch_mlm_lossA = 0
    epoch_mlm_lossB = 0
    epoch_mlm_lossE = 0
    count_A = 0
    count_B = 0
    count_E = 0
    for i, (peptide, alpha, beta, binder, mhc, w) in enumerate(train_dataloader):
        optimizer.zero_grad()
        peptide_input = peptide['input_ids']
        peptide_mask= peptide["attention_mask"]
        peptide_tokentype = peptide['token_type_ids']
        alpha_input = alpha['input_ids']
        alpha_mask = alpha["attention_mask"]
        alpha_tokentype = alpha['token_type_ids']
        beta_input = beta['input_ids']
        beta_mask = beta["attention_mask"]
        beta_tokentype = beta['token_type_ids']
        
        binder = binder
        positive_mask = binder == 1
        alpha_observed_mask = alpha_input.clone().detach()[:,1] != 4
        beta_observed_mask = beta_input.clone().detach()[:,1] != 4
        peptide_observed_mask = peptide_input.clone().detach()[:,1] != 4
        # lm_labels = alphabeta_output.clone()
        clf_label = binder.clone()
        labels = clf_label

        out = model(input_ids=(alpha_input,beta_input,peptide_input), attention_mask=(alpha_mask,beta_mask,peptide_mask),
                                        labels=labels, mhc=mhc)

        loss_clf =torch.mean(torch.mul(out.loss, w)) #decoder_outputs.lossCLS

        lm_lossA = out.decoder_outputsA.lm_loss
        lm_lossB = out.decoder_outputsB.lm_loss
        lm_lossE = out.decoder_outputsE.lm_loss
        # clf_lossA = out.decoder_outputsA.lossCLS
        # clf_lossB = out.decoder_outputsB.lossCLS
        # clf_lossE = out.decoder_outputsE.lossCLS
        prediction_scoresA = out.decoder_outputsA.lm_logits
        predictionsA = F.log_softmax(prediction_scoresA, dim=2)
        prediction_scoresB = out.decoder_outputsB.lm_logits
        predictionsB = F.log_softmax(prediction_scoresB, dim=2)
        prediction_scoresE = out.decoder_outputsE.lm_logits
        predictionsE = F.log_softmax(prediction_scoresE, dim=2)
        lossa = compute_loss(predictionsA[alpha_observed_mask&positive_mask], alpha_input[alpha_observed_mask&positive_mask], criterion) 
        lossb = compute_loss(predictionsB[beta_observed_mask&positive_mask], beta_input[beta_observed_mask&positive_mask], criterion) 
        losse = compute_loss(predictionsE[peptide_observed_mask&positive_mask], peptide_input[peptide_observed_mask&positive_mask], criterion)
        # print(lossa, lossb, losse)
        mlm_lossA = MLM_Loss(model.encoderA, model.MLMHeadA, masker, alpha_input, alpha_mask, alpha_observed_mask)
        mlm_lossB = MLM_Loss(model.encoderB, model.MLMHeadB, masker, beta_input, beta_mask, beta_observed_mask)
        mlm_lossE = MLM_Loss(model.encoderE, model.MLMHeadE, masker, peptide_input, peptide_mask, peptide_observed_mask)

        loss = mlm_lossA+mlm_lossB+mlm_lossE+lossa+lossb+losse+alph*loss_clf
        # loss = lossa+lossb+losse+alph*loss_clf
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        count_A += sum(alpha_observed_mask&positive_mask)
        count_B += sum(beta_observed_mask&positive_mask)
        count_E += sum(peptide_observed_mask&positive_mask)
        epoch_lm_lossA += lossa
        epoch_lm_lossB += lossb
        epoch_lm_lossE += losse
        # epoch_mlm_lossA += mlm_lossA
        # epoch_mlm_lossB += mlm_lossB
        # epoch_mlm_lossE += mlm_lossE
        epoch_clfloss += loss_clf

    epoch_lm_lossA /= count_A
    epoch_lm_lossB /= count_B
    epoch_lm_lossE /= count_E
    epoch_mlm_lossA /= count_A
    epoch_mlm_lossB /= count_B
    epoch_mlm_lossE /= count_E
    epoch_clfloss /= len(train_dataloader.dataset)

    return epoch_lm_lossA, epoch_lm_lossB, epoch_lm_lossE, epoch_mlm_lossA, epoch_mlm_lossB, epoch_mlm_lossE, epoch_clfloss

















def eval_unsupervised(model, masker, test_dataloader, criterion):
    model.eval()
    with torch.no_grad():
        # epoch_loss = 0
        # epoch_clfloss=0
        epoch_lm_lossA = 0
        epoch_lm_lossB = 0
        epoch_lm_lossE = 0
        epoch_mlm_lossA = 0
        epoch_mlm_lossB = 0
        epoch_mlm_lossE = 0
        count_A = 0
        count_B = 0
        count_E = 0
        for i, (peptide, alpha, beta, binder, mhc) in enumerate(test_dataloader):
            peptide_input = peptide['input_ids']
            peptide_mask= peptide["attention_mask"]
            peptide_tokentype = peptide['token_type_ids']
            alpha_input = alpha['input_ids']
            alpha_mask = alpha["attention_mask"]
            alpha_tokentype = alpha['token_type_ids']
            beta_input = beta['input_ids']
            beta_mask = beta["attention_mask"]
            beta_tokentype = beta['token_type_ids']
            
            # binder = binder
            # positive_mask = binder == 1
            alpha_observed_mask = torch.tensor(alpha_input)[:,1] != 4
            beta_observed_mask = torch.tensor(beta_input)[:,1] != 4
            peptide_observed_mask = torch.tensor(peptide_input)[:,1] != 4
            # lm_labels = alphabeta_output.clone()
            clf_label = binder.clone()
            labels = clf_label

            out = model(input_ids=(alpha_input,beta_input,peptide_input), attention_mask=(alpha_mask,beta_mask,peptide_mask),
                                            labels=labels, mhc=mhc)

            loss_clf = out.loss #decoder_outputs.lossCLS
            lm_lossA = out.decoder_outputsA.lm_loss
            lm_lossB = out.decoder_outputsB.lm_loss
            lm_lossE = out.decoder_outputsE.lm_loss
            # clf_lossA = out.decoder_outputsA.lossCLS
            # clf_lossB = out.decoder_outputsB.lossCLS
            # clf_lossE = out.decoder_outputsE.lossCLS
            prediction_scoresA = out.decoder_outputsA.lm_logits
            predictionsA = F.log_softmax(prediction_scoresA, dim=2)
            prediction_scoresB = out.decoder_outputsB.lm_logits
            predictionsB = F.log_softmax(prediction_scoresB, dim=2)
            prediction_scoresE = out.decoder_outputsE.lm_logits
            predictionsE = F.log_softmax(prediction_scoresE, dim=2)
            lossa = compute_loss(predictionsA[alpha_observed_mask], alpha_input[alpha_observed_mask], criterion) 
            lossb = compute_loss(predictionsB[beta_observed_mask], beta_input[beta_observed_mask], criterion) 
            losse = compute_loss(predictionsE[peptide_observed_mask], peptide_input[peptide_observed_mask], criterion)
            mlm_lossA = MLM_Loss(model.encoderA, model.MLMHeadA, masker, alpha_input, alpha_mask, alpha_observed_mask)
            mlm_lossB = MLM_Loss(model.encoderB, model.MLMHeadB, masker, beta_input, beta_mask, beta_observed_mask)

            mlm_lossE = MLM_Loss(model.encoderE, model.MLMHeadE, masker, peptide_input, peptide_mask, peptide_observed_mask)
            # print("mlm", mlm_lossA, mlm_lossB, mlm_lossE)


            # if loss !=0:
            #   epoch_loss += loss.item()
            # loss = alph*loss + (1-alph)*loss_clf
            # if loss ==0 and alph ==1:
            #     epoch_clfloss += loss_clf.item()
            #     continue
            loss = mlm_lossA+mlm_lossB+mlm_lossE+lossa+lossb+losse


            # print(model.decoder.bert.pooler.dense.weight)
            count_A += sum(alpha_observed_mask)
            count_B += sum(beta_observed_mask)
            count_E += sum(peptide_observed_mask)
            epoch_lm_lossA += lossa
            epoch_lm_lossB += lossb
            epoch_lm_lossE += losse
            epoch_mlm_lossA += mlm_lossA
            epoch_mlm_lossB += mlm_lossB
            epoch_mlm_lossE += mlm_lossE

        epoch_lm_lossA /= count_A
        epoch_lm_lossB /= count_B
        epoch_lm_lossE /= count_E
        epoch_mlm_lossA /= count_A
        epoch_mlm_lossB /= count_B
        epoch_mlm_lossE /= count_E

        return epoch_lm_lossA, epoch_lm_lossB, epoch_lm_lossE, epoch_mlm_lossA, epoch_mlm_lossB, epoch_mlm_lossE


def eval_supervised(model, masker, test_dataloader, criterion):
    model.eval()
    with torch.no_grad():
        epoch_clfloss=0
        epoch_lm_lossA = 0
        epoch_lm_lossB = 0
        epoch_lm_lossE = 0
        epoch_mlm_lossA = 0
        epoch_mlm_lossB = 0
        epoch_mlm_lossE = 0
        count_A = 0
        count_B = 0
        count_E = 0
        clf_score = []
        Boolbinders = []
        for i, (peptide, alpha, beta, binder, mhc) in enumerate(test_dataloader):
            peptide_input = peptide['input_ids']
            peptide_mask= peptide["attention_mask"]
            peptide_tokentype = peptide['token_type_ids']
            alpha_input = alpha['input_ids']
            alpha_mask = alpha["attention_mask"]
            alpha_tokentype = alpha['token_type_ids']
            beta_input = beta['input_ids']
            beta_mask = beta["attention_mask"]
            beta_tokentype = beta['token_type_ids']

            binder = binder
            positive_mask = binder == 1
            alpha_observed_mask = alpha_input.clone().detach()[:,1] != 4
            beta_observed_mask = beta_input.clone().detach()[:,1] != 4
            peptide_observed_mask = peptide_input.clone().detach()[:,1] != 4
            # lm_labels = alphabeta_output.clone()
            clf_label = binder.clone()
            labels = clf_label

            out = model(input_ids=(alpha_input,beta_input,peptide_input), attention_mask=(alpha_mask,beta_mask,peptide_mask),
                                            labels=labels, mhc=mhc)

            def softm(x):
                x = torch.exp(x)
                su = torch.sum(x, dim=1)
                # print(x.shape, su.shape)
                x = x/su
                return x

            clf_logits = softm(out.clf_logits)
            clf_score.append(clf_logits[0,1].cpu().item())
            Boolbinders.append((binder[0]==1).cpu().item())
            lm_lossA = out.decoder_outputsA.lm_loss
            lm_lossB = out.decoder_outputsB.lm_loss
            lm_lossE = out.decoder_outputsE.lm_loss
            # clf_lossA = out.decoder_outputsA.lossCLS
            # clf_lossB = out.decoder_outputsB.lossCLS
            # clf_lossE = out.decoder_outputsE.lossCLS
            prediction_scoresA = out.decoder_outputsA.lm_logits
            predictionsA = F.log_softmax(prediction_scoresA, dim=2)
            prediction_scoresB = out.decoder_outputsB.lm_logits
            predictionsB = F.log_softmax(prediction_scoresB, dim=2)
            prediction_scoresE = out.decoder_outputsE.lm_logits
            predictionsE = F.log_softmax(prediction_scoresE, dim=2)
            lossa = compute_loss(predictionsA[alpha_observed_mask&positive_mask], alpha_input[alpha_observed_mask&positive_mask], criterion) 
            lossb = compute_loss(predictionsB[beta_observed_mask&positive_mask], beta_input[beta_observed_mask&positive_mask], criterion) 
            losse = compute_loss(predictionsE[peptide_observed_mask&positive_mask], peptide_input[peptide_observed_mask&positive_mask], criterion)

            mlm_lossA = MLM_Loss(model.encoderA, model.MLMHeadA, masker, alpha_input, alpha_mask, alpha_observed_mask)
            mlm_lossB = MLM_Loss(model.encoderB, model.MLMHeadB, masker, beta_input, beta_mask, beta_observed_mask)
            mlm_lossE = MLM_Loss(model.encoderE, model.MLMHeadE, masker, peptide_input, peptide_mask, peptide_observed_mask)


            count_A += sum(alpha_observed_mask&positive_mask)
            count_B += sum(beta_observed_mask&positive_mask)
            count_E += sum(peptide_observed_mask&positive_mask)
            epoch_lm_lossA += lossa
            epoch_lm_lossB += lossb
            epoch_lm_lossE += losse
            epoch_mlm_lossA += mlm_lossA
            epoch_mlm_lossB += mlm_lossB
            epoch_mlm_lossE += mlm_lossE
            # epoch_clfloss += loss_clf

        epoch_lm_lossA /= count_A
        epoch_lm_lossB /= count_B
        epoch_lm_lossE /= count_E
        epoch_mlm_lossA /= count_A
        epoch_mlm_lossB /= count_B
        epoch_mlm_lossE /= count_E
        epoch_clfloss/=len(test_dataloader.dataset)
        auc2 = roc_auc_score(Boolbinders, clf_score)

    return epoch_lm_lossA, epoch_lm_lossB, epoch_lm_lossE, epoch_mlm_lossA, epoch_mlm_lossB, epoch_mlm_lossE, epoch_clfloss, auc2


























def eval_model_old(model, optimizer, valid_dataloader):

    model.eval()
    epoch_loss = 0
    prob_perProt = []
    loss_binding = []
    Boolbinders = []
    clf_score = []
    for i, (peptide, alpha, beta, binder, mhc) in enumerate(valid_dataloader):
        peptide_input = peptide['input_ids']
        peptide_mask= peptide["attention_mask"]
        peptide_tokentype = peptide['token_type_ids']
        
        alpha_input = alpha['input_ids']
        alpha_mask = alpha["attention_mask"]
        alpha_tokentype = alpha['token_type_ids']

        beta_input = beta['input_ids']
        beta_mask = beta["attention_mask"]
        beta_tokentype = beta['token_type_ids']
        
        binder = binder.to(device)
        positive_mask = binder == 1
        alpha_observed_mask = torch.tensor(alpha_input).clone().detach()[:,1] != 4
        beta_observed_mask = torch.tensor(beta_input)[:,1] != 4

        clf_label = binder.clone()
        labels = (None)

        out = model(input_ids=(alpha_input,beta_input,peptide_input), attention_mask=(alpha_mask,beta_mask,peptide_mask),
                                        labels=labels)

        def softm(x):
            x = torch.exp(x)
            su = torch.sum(x, dim=1)
            x = x/su
            return x
        clf_logits = softm(out.clf_logits)
        clf_score.append(clf_logits[0,1].cpu().item())
        prediction_scoresA = out.decoder_outputsA.lm_logits
        predictionsA = F.log_softmax(prediction_scoresA, dim=2)
        prediction_scoresB = out.decoder_outputsB.lm_logits
        predictionsB = F.log_softmax(prediction_scoresB, dim=2)
        prediction_scoresE = out.decoder_outputsE.lm_logits
        predictionsE = F.log_softmax(prediction_scoresE, dim=2)
        loss = compute_loss(predictionsA[alpha_observed_mask & positive_mask], alpha_input[alpha_observed_mask & positive_mask]) + compute_loss(predictionsB[beta_observed_mask & positive_mask], beta_input[beta_observed_mask&positive_mask]) + compute_loss(predictionsE[positive_mask], peptide_input[positive_mask])

        Boolbinders.append((binder[0]==1).cpu().item())
        if binder[0]==1:
            epoch_loss += loss.item()
    # print(len(Boolbinders),len(clf_score))
    # auc = roc_auc_score(Boolbinders, prob_perProt)
    auc2 = roc_auc_score(Boolbinders, clf_score)
    print("Mean validation loss:", (epoch_loss / num_valid_batches),  "auc2", auc2)






def train_model_old(model, optimizer, train_dataloader, alph=0.5):
    model.train()
    epoch_loss = 0
    epoch_clfloss=0
    for i, (peptide, alpha, beta, binder) in enumerate(train_dataloader):
    #for i, batch in enumerate(train_dataloader):
#         batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()

        peptide_input = peptide['input_ids']
        peptide_mask= peptide["attention_mask"]
        peptide_tokentype = peptide['token_type_ids']
        
        alpha_input = alpha['input_ids']
        alpha_mask = alpha["attention_mask"]
        alpha_tokentype = alpha['token_type_ids']

        beta_input = beta['input_ids']
        beta_mask = beta["attention_mask"]
        beta_tokentype = beta['token_type_ids']
        
        binder = binder
        positive_mask = binder == 1
        alpha_observed_mask = torch.tensor(alpha_input)[:,1] != 4
        beta_observed_mask = torch.tensor(beta_input)[:,1] != 4

        # lm_labels = alphabeta_output.clone()
        clf_label = binder.clone()
        labels = clf_label

        out = model(input_ids=(alpha_input,beta_input,peptide_input), attention_mask=(alpha_mask,beta_mask,peptide_mask),
                                        labels=labels)

        loss_clf = out.loss#decoder_outputs.lossCLS
        lm_lossA = out.decoder_outputsA.lm_loss
        lm_lossB = out.decoder_outputsB.lm_loss
        lm_lossE = out.decoder_outputsE.lm_loss
        clf_lossA = out.decoder_outputsA.lossCLS
        clf_lossB = out.decoder_outputsB.lossCLS
        clf_lossE = out.decoder_outputsE.lossCLS
        prediction_scoresA = out.decoder_outputsA.lm_logits
        predictionsA = F.log_softmax(prediction_scoresA, dim=2)
        prediction_scoresB = out.decoder_outputsB.lm_logits
        predictionsB = F.log_softmax(prediction_scoresB, dim=2)
        prediction_scoresE = out.decoder_outputsE.lm_logits
        predictionsE = F.log_softmax(prediction_scoresE, dim=2)
        loss = compute_loss(predictionsA[alpha_observed_mask & positive_mask], alpha_input[alpha_observed_mask & positive_mask]) + compute_loss(predictionsB[beta_observed_mask & positive_mask], beta_input[beta_observed_mask&positive_mask]) + compute_loss(predictionsE[positive_mask], peptide_input[positive_mask])

        if loss !=0:
          epoch_loss += loss.item()
        loss = alph*loss + (1-alph)*loss_clf
        if loss ==0 and alph ==1:
            epoch_clfloss += loss_clf.item()
            continue

        loss.backward()
        # loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # print(loss_clf)#model.decoder.pooler.dense.weight)
        optimizer.step()
        # print(model.decoder.bert.pooler.dense.weight)
        epoch_clfloss += loss_clf.item()

    print("Mean epoch loss:", (epoch_loss / num_train_batches), (epoch_clfloss / num_train_batches))

def eval_model_old(model, optimizer, valid_dataloader):

    model.eval()
    epoch_loss = 0
    prob_perProt = []
    loss_binding = []
    Boolbinders = []
    clf_score = []
    for i, (peptide, alpha, beta, binder) in enumerate(valid_dataloader):
        peptide_input = peptide['input_ids']
        peptide_mask= peptide["attention_mask"]
        peptide_tokentype = peptide['token_type_ids']
        
        alpha_input = alpha['input_ids']
        alpha_mask = alpha["attention_mask"]
        alpha_tokentype = alpha['token_type_ids']

        beta_input = beta['input_ids']
        beta_mask = beta["attention_mask"]
        beta_tokentype = beta['token_type_ids']
        
        binder = binder.to(device)
        positive_mask = binder == 1
        alpha_observed_mask = torch.tensor(alpha_input)[:,1] != 4
        beta_observed_mask = torch.tensor(beta_input)[:,1] != 4

        clf_label = binder.clone()
        labels = (None)

        out = model(input_ids=(alpha_input,beta_input,peptide_input), attention_mask=(alpha_mask,beta_mask,peptide_mask),
                                        labels=labels)

        def softm(x):
            x = torch.exp(x)
            su = torch.sum(x, dim=1)
            x = x/su
            return x
        clf_logits = softm(out.clf_logits)
        clf_score.append(clf_logits[0,1].cpu().item())
        prediction_scoresA = out.decoder_outputsA.lm_logits
        predictionsA = F.log_softmax(prediction_scoresA, dim=2)
        prediction_scoresB = out.decoder_outputsB.lm_logits
        predictionsB = F.log_softmax(prediction_scoresB, dim=2)
        prediction_scoresE = out.decoder_outputsE.lm_logits
        predictionsE = F.log_softmax(prediction_scoresE, dim=2)
        loss = compute_loss(predictionsA[alpha_observed_mask & positive_mask], alpha_input[alpha_observed_mask & positive_mask]) + compute_loss(predictionsB[beta_observed_mask & positive_mask], beta_input[beta_observed_mask&positive_mask]) + compute_loss(predictionsE[positive_mask], peptide_input[positive_mask])

        Boolbinders.append((binder[0]==1).cpu().item())
        if binder[0]==1:
            epoch_loss += loss.item()
    # print(len(Boolbinders),len(clf_score))
    # auc = roc_auc_score(Boolbinders, prob_perProt)
    auc2 = roc_auc_score(Boolbinders, clf_score)
    print("Mean validation loss:", (epoch_loss / num_valid_batches),  "auc2", auc2)







class MyMasking():
    def __init__(self,tokenizer, mlm_probability: float = 0.15):
        self.tokenizer=tokenizer
        self.mlm_probability=mlm_probability

    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
            """
            Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
            """
            import torch
            labels = inputs.clone()
            # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
            probability_matrix = torch.full(labels.shape, self.mlm_probability)
            if special_tokens_mask is None:
                special_tokens_mask = [
                    self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
                ]
                special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
            else:
                special_tokens_mask = special_tokens_mask.bool()

            probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
            masked_indices = torch.bernoulli(probability_matrix).bool()
            labels[~masked_indices] = -100  # We only compute loss on masked tokens

            # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
            indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
            inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

            # 10% of the time, we replace masked input tokens with random word
            indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
            random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long).to(inputs.device)
            inputs[indices_random] = random_words[indices_random]

            # The rest of the time (10% of the time) we keep the masked input tokens unchanged
            return inputs, labels












# tokenizer = AutoTokenizer.from_pretrained("dataTrans/")#lightonai/RITA_l")#/content/drive/MyDrive/phd/TCREp/")
# if tokenizer.pad_token is None:
#     tokenizer.add_special_tokens({'pad_token': '<PAD>'})

# if tokenizer.sep_token is None:
#     tokenizer.add_special_tokens({'sep_token': '<MIS>'})
    
# if tokenizer.cls_token is None:
#     tokenizer.add_special_tokens({'cls_token': '<CLS>'})

# if tokenizer.eos_token is None:
#     tokenizer.add_special_tokens({'eos_token': '<EOS>'})

# from tokenizers.processors import TemplateProcessing
# tokenizer._tokenizer.post_processor = TemplateProcessing(
#     single="<CLS> $A <EOS>",
#     pair="<CLS> $A <MIS> $B:1 <EOS>:1",
#     special_tokens=[
#         ("<EOS>", 2),
#         ("<CLS>", 3),
#         ("<MIS>", 4),
#     ],
# )






# datasetTrain = TCRDataset("dataTrans/VDJ_train.csv", tokenizer, target_binder=1)
# datasetValid = TCRDataset("dataTrans/VDJ_test.csv", tokenizer)

# print(len(datasetValid))
# train_dataloader = torch.utils.data.DataLoader(dataset=datasetTrain, batch_size=64, shuffle=False, collate_fn=datasetTrain.ab2e_collate_function) 
# #                                                 drop_last=False, num_workers=1, collate_fn=dataset.ab2e_collate_function)


# valid_dataloader = torch.utils.data.DataLoader(dataset=datasetValid, batch_size=1, shuffle=False, collate_fn=datasetValid.ab2e_collate_function) 
# #                                                 drop_last=False, num_workers=1, collate_fn=dataset.ab2e_collate_function)

# num_train_batches = len(train_dataloader)
# num_valid_batches = np.sum(np.array(valid_dataloader.dataset.binder) == 1) #len(valid_dataloader)
# num_valid_batches



def LLLoss_raw(predictions, targets, ignore_index):
    """Compute our custom loss"""
    criterion = nn.NLLLoss(ignore_index=ignore_index, reduction='none')
    if len(targets)>0:
      predictions = predictions[:, :-1, :].contiguous()
      targets = targets[:, 1:]
      bs = targets.shape[0]

      rearranged_output = predictions.view(predictions.shape[0]*predictions.shape[1], -1)
      rearranged_target = targets.contiguous().view(-1)

      loss = criterion(rearranged_output, rearranged_target).reshape(bs,-1).sum(dim=1)
    else:
      loss = torch.zeros(1)
    return loss



def unsupervised_auc(model, test_dataloader, ignore_index):
    model.eval()
    with torch.no_grad():

        clf_scorea = []
        clf_scoreb = []
        clf_scoree = []
        Boolbinders = []
        for i, (peptide, alpha, beta, binder, mhc) in enumerate(test_dataloader):
            peptide_input = peptide['input_ids']
            peptide_mask= peptide["attention_mask"]
            peptide_tokentype = peptide['token_type_ids']
            alpha_input = alpha['input_ids']
            alpha_mask = alpha["attention_mask"]
            alpha_tokentype = alpha['token_type_ids']
            beta_input = beta['input_ids']
            beta_mask = beta["attention_mask"]
            beta_tokentype = beta['token_type_ids']

            binder = binder
            positive_mask = binder == 1
            alpha_observed_mask = alpha_input.clone().detach()[:,1] != 4
            beta_observed_mask = beta_input.clone().detach()[:,1] != 4
            peptide_observed_mask = peptide_input.clone().detach()[:,1] != 4

            clf_label = binder.clone()
            labels = clf_label

            out = model(input_ids=(alpha_input,beta_input,peptide_input), attention_mask=(alpha_mask,beta_mask,peptide_mask),
                                            labels=labels, mhc=mhc)

            def softm(x):
                x = torch.exp(x)
                su = torch.sum(x, dim=1)
                x = x/su
                return x

            Boolbinders += [(binder[i]==1).cpu().item() for i in range(len(binder))]
            lm_lossA = out.decoder_outputsA.lm_loss
            lm_lossB = out.decoder_outputsB.lm_loss
            lm_lossE = out.decoder_outputsE.lm_loss

            prediction_scoresA = out.decoder_outputsA.lm_logits
            predictionsA = F.log_softmax(prediction_scoresA, dim=2)
            prediction_scoresB = out.decoder_outputsB.lm_logits
            predictionsB = F.log_softmax(prediction_scoresB, dim=2)
            prediction_scoresE = out.decoder_outputsE.lm_logits
            predictionsE = F.log_softmax(prediction_scoresE, dim=2)

            lossa = LLLoss_raw(predictionsA, alpha_input, ignore_index) 
            lossb = LLLoss_raw(predictionsB, beta_input, ignore_index) 
            losse = LLLoss_raw(predictionsE, peptide_input, ignore_index)
            # print(lossa.shape, alpha_input.shape, lossb.shape, beta_input.shape, losse.shape )
            clf_scorea += [lossa[i].cpu().item() for i in range(len(lossa))]
            clf_scoreb += [lossb[i].cpu().item() for i in range(len(lossb))]
            clf_scoree += [losse[i].cpu().item() for i in range(len(losse))]

        # print(Boolbinders)
        # print(clf_scoree)
        auca = roc_auc_score(Boolbinders, clf_scorea)
        aucb = roc_auc_score(Boolbinders, clf_scoreb)
        auce = roc_auc_score(Boolbinders, clf_scoree)
        return auca, aucb, auce





def supervised_auc(model, test_dataloader, ignore_index):
    model.eval()
    with torch.no_grad():

        clf_scorea = []
        clf_scoreb = []
        clf_scoree = []
        Boolbinders = []
        for i, (peptide, alpha, beta, binder, mhc) in enumerate(test_dataloader):
            peptide_input = peptide['input_ids']
            peptide_mask= peptide["attention_mask"]
            peptide_tokentype = peptide['token_type_ids']
            alpha_input = alpha['input_ids']
            alpha_mask = alpha["attention_mask"]
            alpha_tokentype = alpha['token_type_ids']
            beta_input = beta['input_ids']
            beta_mask = beta["attention_mask"]
            beta_tokentype = beta['token_type_ids']

            binder = binder
            positive_mask = binder == 1
            alpha_observed_mask = alpha_input.clone().detach()[:,1] != 4
            beta_observed_mask = beta_input.clone().detach()[:,1] != 4
            peptide_observed_mask = peptide_input.clone().detach()[:,1] != 4

            clf_label = binder.clone()
            labels = clf_label

            out = model(input_ids=(alpha_input,beta_input,peptide_input), attention_mask=(alpha_mask,beta_mask,peptide_mask),
                                            labels=labels, mhc=mhc)

            def softm(x):
                x = torch.exp(x)
                su = torch.sum(x, dim=1)
                x = x/su
                return x

            Boolbinders += [(binder[i]==1).cpu().item() for i in range(len(binder))]
            lm_lossA = out.decoder_outputsA.lm_loss
            lm_lossB = out.decoder_outputsB.lm_loss
            lm_lossE = out.decoder_outputsE.lm_loss

            prediction_scoresA = out.decoder_outputsA.lm_logits
            predictionsA = F.log_softmax(prediction_scoresA, dim=2)
            prediction_scoresB = out.decoder_outputsB.lm_logits
            predictionsB = F.log_softmax(prediction_scoresB, dim=2)
            prediction_scoresE = out.decoder_outputsE.lm_logits
            predictionsE = F.log_softmax(prediction_scoresE, dim=2)

            lossa = LLLoss_raw(predictionsA, alpha_input, ignore_index) 
            lossb = LLLoss_raw(predictionsB, beta_input, ignore_index) 
            losse = LLLoss_raw(predictionsE, peptide_input, ignore_index)
            # print(lossa.shape, alpha_input.shape, lossb.shape, beta_input.shape, losse.shape )
            clf_scorea += [lossa[i].cpu().item()+lossb[i].cpu().item() for i in range(len(lossa))]
            clf_scoreb += [lossb[i].cpu().item()+lossb[i].cpu().item() for i in range(len(lossb))]
            clf_scoree += [losse[i].cpu().item() for i in range(len(losse))]

        auca = roc_auc_score(Boolbinders, clf_scorea)
        aucb = roc_auc_score(Boolbinders, clf_scoreb)
        auce = roc_auc_score(Boolbinders, clf_scoree)
        return auca, aucb, auce


####

def meanLLdataset(model, test_dataloader, ignore_index):
    model.eval()
    with torch.no_grad():

        clf_scorea = []
        clf_scoreb = []
        clf_scoree = []
        Boolbinders = []
        for i, (peptide, alpha, beta, binder, mhc) in enumerate(test_dataloader):
            peptide_input = peptide['input_ids']
            peptide_mask= peptide["attention_mask"]
            peptide_tokentype = peptide['token_type_ids']
            alpha_input = alpha['input_ids']
            alpha_mask = alpha["attention_mask"]
            alpha_tokentype = alpha['token_type_ids']
            beta_input = beta['input_ids']
            beta_mask = beta["attention_mask"]
            beta_tokentype = beta['token_type_ids']

            binder = binder
            positive_mask = binder == 1
            alpha_observed_mask = alpha_input.clone().detach()[:,1] != 4
            beta_observed_mask = beta_input.clone().detach()[:,1] != 4
            peptide_observed_mask = peptide_input.clone().detach()[:,1] != 4

            clf_label = binder.clone()
            labels = clf_label

            out = model(input_ids=(alpha_input,beta_input,peptide_input), attention_mask=(alpha_mask,beta_mask,peptide_mask),
                                            labels=labels, mhc=mhc)

            def softm(x):
                x = torch.exp(x)
                su = torch.sum(x, dim=1)
                x = x/su
                return x

            Boolbinders += [(binder[i]==1).cpu().item() for i in range(len(binder))]
            lm_lossA = out.decoder_outputsA.lm_loss
            lm_lossB = out.decoder_outputsB.lm_loss
            lm_lossE = out.decoder_outputsE.lm_loss

            prediction_scoresA = out.decoder_outputsA.lm_logits
            predictionsA = F.log_softmax(prediction_scoresA, dim=2)
            prediction_scoresB = out.decoder_outputsB.lm_logits
            predictionsB = F.log_softmax(prediction_scoresB, dim=2)
            prediction_scoresE = out.decoder_outputsE.lm_logits
            predictionsE = F.log_softmax(prediction_scoresE, dim=2)

            lossa = LLLoss_raw(predictionsA, alpha_input, ignore_index) 
            lossb = LLLoss_raw(predictionsB, beta_input, ignore_index) 
            losse = LLLoss_raw(predictionsE, peptide_input, ignore_index)
            # print(lossa.shape, alpha_input.shape, lossb.shape, beta_input.shape, losse.shape )
            clf_scorea += [lossa[i].cpu().item() for i in range(len(lossa))]
            clf_scoreb += [lossb[i].cpu().item() for i in range(len(lossb))]
            clf_scoree += [losse[i].cpu().item() for i in range(len(losse))]

        # print(Boolbinders)
        # print(clf_scoree)
        # auca = roc_auc_score(Boolbinders, clf_scorea)
        # aucb = roc_auc_score(Boolbinders, clf_scoreb)
        # auce = roc_auc_score(Boolbinders, clf_scoree)
        scorea = np.mean(clf_scorea)
        scoreb = np.mean(clf_scoreb)
        scoree = np.mean(clf_scoree)
        return scorea, scoreb, scoree


























# def ReyniMatchingLossNew(batch,
#                                     model,
#                                     criterion,
#                                     criterionMatching,
#                                     device,
#                                     accumulate=False,
#                                     ncontrastive=5,
#                                     sampler="gumbel"):

#     inp_data, target, idx_list = batch[0], batch[1], batch[2]
#     bs = inp_data.shape[1]

        
#     lossMatrix = torch.zeros((bs,ncontrastive+1)).to(device)
#     LossCE = torch.tensor(0.0).to(device)
#     lossMatching = torch.tensor(0.0).to(device)
#     targetMatching = torch.tensor([0]*bs).to(device)
    
    
    
#     output = model(inp_data, target[:-1, :])
#     output = output.reshape(-1, output.shape[2])
#     targets_Original = target
#     targets_Original = targets_Original[1:].reshape(-1)
#     loss = criterion(output, targets_Original).reshape(-1,bs).mean(dim=0)
#     lossMatrix[:,0] = loss
#     LossCE += loss.mean()
    
#     for i in range(1,ncontrastive+1):
#         contrastiveTarget = model.pseudosample(inp_data, target, nsample=1, method=sampler)
#         output2 = model(inp_data, contrastiveTarget[:-1, :])
#         output2 = output2.reshape(-1, output2.shape[2])
#         _, targets_Original2 = contrastiveTarget.max(dim=2)
#         targets_Original2 = targets_Original2[1:].reshape(-1)
#         loss2 = criterion(output2, targets_Original2).reshape(-1,bs).mean(dim=0)
#         lossMatrix[:,i] = loss2
        
#     lossMatrix *=-1
#     lossMatching = -1*criterionMatching(lossMatrix, targetMatching)

#     return LossCE, lossMatching


#   def pseudosample(model, model_output)
#         model_output.decoder_outputsA.logits



#         outputs = torch.zeros(target.shape[0], nsample, self.trg_vocab_size).to(self.device)
#         outputs[0,:,:] = sos.unsqueeze(0).repeat(nsample, 1)
#         output = self.forward(inp, target[:-1, :])
#         best_guess = torch.nn.functional.gumbel_softmax(output, hard=True, dim=2)
#         #print("outputs shape", outputs.shape, "bg", best_guess.shape)
#         outputs[1:,:,:] = best_guess
#         outputs[-1,:,:] = eos.unsqueeze(0).repeat(nsample, 1)
#         # if method=="gumbel":
#         #     if self.onehot:
#         #         outputs = torch.zeros(target.shape[0], nsample, target.shape[2]).to(self.device)
#         #         outputs[0,:,:] = sos.unsqueeze(0).repeat(nsample, 1)
    
#         #         output = self.forward(inp, target[:-1, :])
#         #         best_guess = torch.nn.functional.gumbel_softmax(output, hard=True, dim=2)
#         #         outputs[1:,:,:]= best_guess
#         #         outputs[-1,:,:] = eos.unsqueeze(0).repeat(nsample, 1)
#             else:



# train_dataloader_pretune = torch.utils.data.DataLoader(dataset=datasetTrain_pretune, batch_size=12, shuffle=True, collate_fn=datasetValidFinal.all2allmhc_collate_function) 

# batch=next(iter(train_dataloader_pretune))


# (peptide, alpha, beta, binder, mhc) = batch
# peptide_input = peptide['input_ids']
# peptide_mask= peptide["attention_mask"]
# peptide_tokentype = peptide['token_type_ids']
# alpha_input = alpha['input_ids']
# alpha_mask = alpha["attention_mask"]
# alpha_tokentype = alpha['token_type_ids']
# beta_input = beta['input_ids']
# beta_mask = beta["attention_mask"]
# beta_tokentype = beta['token_type_ids']

# binder = binder
# positive_mask = binder == 1
# alpha_observed_mask = alpha_input.clone().detach()[:,1] != 4
# beta_observed_mask = beta_input.clone().detach()[:,1] != 4
# peptide_observed_mask = peptide_input.clone().detach()[:,1] != 4

# clf_label = binder.clone()
# labels = clf_label

# out = model(input_ids=(alpha_input,beta_input,peptide_input), attention_mask=(alpha_mask,beta_mask,peptide_mask),
#                                 labels=labels, mhc=mhc)


# outputE = peptide_input.clone()
# outputE = torch.nn.functional.one_hot(outputE, num_classes=30)
# best_guess = torch.nn.functional.gumbel_softmax(out.decoder_outputsE.lm_logits, hard=True, dim=2)
# outputE[:,1:,:] = best_guess[:,:-1,:]
# outputE = torch.matmul(outputE, model.encoderE.get_input_embeddings().weight)


# encoder_outputsE_reyni = model.encoderE(
#                 input_ids=None,
#                 attention_mask=attention_maskE,
#                 inputs_embeds=outputE,
#                 output_attentions=output_attentions,
#                 output_hidden_states=output_hidden_states,
#                 return_dict=return_dict,
#                 **kwargs_encoder,
#             )


# labelsE = (labels, input_idsE)
# decoder_outputsE_reyni = self.decoderE(
#     input_ids = input_idsE,
#     attention_mask = attention_maskE,
#     encoder_hidden_states = torch.cat([mhc_encoded,encoder_hidden_statesA, encoder_hidden_statesB], dim=1),
#     encoder_attention_mask = torch.cat([mhc_attention_mask,attention_maskA, attention_maskB], dim=1),
#     inputs_embeds = inputs_embedsE,
#     output_attentions = output_attentions,
#     output_hidden_states = output_hidden_states,
#     labels=labelsE,
#     use_cache=use_cache,
#     past_key_values=past_key_valuesE,
#     return_dict=return_dict,
#     **kwargs_decoder,
# )

# for i in range()

# torch.nn.functional.one_hot(peptide_input, num_classes=30)
# checkpoint = torch.load("../NetTCR/multiTCR_s/pytorch_model.bin")
# model.load_state_dict(checkpoint)











def train_unsupervised_Reyni(model, optimizer, masker, train_dataloader, criterion, alph = 1.0, alphareyni=0.1):
    model.train()
    # epoch_loss = 0
    # epoch_clfloss=0
    epoch_lm_lossA = 0
    epoch_lm_lossB = 0
    epoch_lm_lossE = 0
    epoch_mlm_lossA = 0
    epoch_mlm_lossB = 0
    epoch_mlm_lossE = 0
    count_A = 0
    count_B = 0
    count_E = 0
    for i, (peptide, alpha, beta, binder, mhc) in enumerate(train_dataloader):
        optimizer.zero_grad()
        peptide_input = peptide['input_ids']
        peptide_mask= peptide["attention_mask"]
        peptide_tokentype = peptide['token_type_ids']
        alpha_input = alpha['input_ids']
        alpha_mask = alpha["attention_mask"]
        alpha_tokentype = alpha['token_type_ids']
        beta_input = beta['input_ids']
        beta_mask = beta["attention_mask"]
        beta_tokentype = beta['token_type_ids']
        
        # binder = binder
        # positive_mask = binder == 1
        alpha_observed_mask = alpha_input.clone().detach()[:,1] != 4
        beta_observed_mask = beta_input.clone().detach()[:,1] != 4
        peptide_observed_mask = peptide_input.clone().detach()[:,1] != 4
        # lm_labels = alphabeta_output.clone()
        clf_label = binder.clone()
        labels = clf_label

        out = model(input_ids=(alpha_input,beta_input,peptide_input), attention_mask=(alpha_mask,beta_mask,peptide_mask),
                                        labels=labels, mhc=mhc)

        loss_clf = out.loss #decoder_outputs.lossCLS
        lm_lossA = out.decoder_outputsA.lm_loss
        lm_lossB = out.decoder_outputsB.lm_loss
        lm_lossE = out.decoder_outputsE.lm_loss
        # clf_lossA = out.decoder_outputsA.lossCLS
        # clf_lossB = out.decoder_outputsB.lossCLS
        # clf_lossE = out.decoder_outputsE.lossCLS
        prediction_scoresA = out.decoder_outputsA.lm_logits
        predictionsA = F.log_softmax(prediction_scoresA, dim=2)
        prediction_scoresB = out.decoder_outputsB.lm_logits
        predictionsB = F.log_softmax(prediction_scoresB, dim=2)
        prediction_scoresE = out.decoder_outputsE.lm_logits
        predictionsE = F.log_softmax(prediction_scoresE, dim=2)
        lossa = compute_loss(predictionsA[alpha_observed_mask], alpha_input[alpha_observed_mask], criterion) 
        lossb = compute_loss(predictionsB[beta_observed_mask], beta_input[beta_observed_mask],criterion) 
        losse = compute_loss(predictionsE[peptide_observed_mask], peptide_input[peptide_observed_mask], criterion)
        # print("lossa", lossa, lm_lossA)
        # print("lossb", lossb, lm_lossB)
        # print("losse", losse, lm_lossE)
        mlm_lossA = MLM_Loss(model.encoderA, model.MLMHeadA, masker, alpha_input, alpha_mask, alpha_observed_mask)
        mlm_lossB = MLM_Loss(model.encoderB, model.MLMHeadB, masker, beta_input, beta_mask, beta_observed_mask)
        mlm_lossE = MLM_Loss(model.encoderE, model.MLMHeadE, masker, peptide_input, peptide_mask, peptide_observed_mask)


        # if loss !=0:
        #   epoch_loss += loss.item()
        # loss = alph*loss + (1-alph)*loss_clf
        # if loss ==0 and alph ==1:
        #     epoch_clfloss += loss_clf.item()
        #     continue
        
        loss = mlm_lossA+mlm_lossB+mlm_lossE+alph*(lossa+lossb+losse)
        # loss = lossa+lossb+losse

        loss.backward()
        # loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # print(loss_clf)#model.decoder.pooler.dense.weight)
        optimizer.step()
        # print(model.decoder.bert.pooler.dense.weight)
        count_A += sum(alpha_observed_mask)
        count_B += sum(beta_observed_mask)
        count_E += sum(peptide_observed_mask)
        epoch_lm_lossA += lossa
        epoch_lm_lossB += lossb
        epoch_lm_lossE += losse
        epoch_mlm_lossA += mlm_lossA
        epoch_mlm_lossB += mlm_lossB
        epoch_mlm_lossE += mlm_lossE

    epoch_lm_lossA /= count_A
    epoch_lm_lossB /= count_B
    epoch_lm_lossE /= count_E
    epoch_mlm_lossA /= count_A
    epoch_mlm_lossB /= count_B
    epoch_mlm_lossE /= count_E

    return epoch_lm_lossA, epoch_lm_lossB, epoch_lm_lossE, epoch_mlm_lossA, epoch_mlm_lossB, epoch_mlm_lossE






############################
#########################3
#########################

def LLLoss_raw2(predictions, targets, ignore_index):
    """Compute our custom loss"""
    criterion = nn.NLLLoss(ignore_index=ignore_index, reduction='none')
    if len(targets)>0:
      predictions = predictions[:, :-1, :].contiguous()
      targets = targets[:, 1:]
      bs = targets.shape[0]
      rearranged_output = predictions.view(predictions.shape[0]*predictions.shape[1], -1)
      rearranged_target = targets.contiguous().view(-1)
      loss = criterion(rearranged_output, rearranged_target).reshape(bs,-1)#.sum(dim=1)
    else:
      loss = torch.zeros(1)
    return loss




def unsupervised_auc2(model, test_dataloader, ignore_index):
    model.eval()
    with torch.no_grad():
        clf_scorea = []
        clf_scoreb = []
        clf_scoree = []
        Boolbinders = []
        for i, (peptide, alpha, beta, binder, mhc) in enumerate(test_dataloader):
            peptide_input = peptide['input_ids']
            peptide_mask= peptide["attention_mask"]
            peptide_tokentype = peptide['token_type_ids']
            alpha_input = alpha['input_ids']
            alpha_mask = alpha["attention_mask"]
            alpha_tokentype = alpha['token_type_ids']
            beta_input = beta['input_ids']
            beta_mask = beta["attention_mask"]
            beta_tokentype = beta['token_type_ids']
            binder = binder
            positive_mask = binder == 1
            alpha_observed_mask = alpha_input.clone().detach()[:,1] != 4
            beta_observed_mask = beta_input.clone().detach()[:,1] != 4
            peptide_observed_mask = peptide_input.clone().detach()[:,1] != 4
            clf_label = binder.clone()
            labels = clf_label
            out = model(input_ids=(alpha_input,beta_input,peptide_input), attention_mask=(alpha_mask,beta_mask,peptide_mask),
                                            labels=labels, mhc=mhc)
            def softm(x):
                x = torch.exp(x)
                su = torch.sum(x, dim=1)
                x = x/su
                return x
            Boolbinders += [(binder[i]==1).cpu().item() for i in range(len(binder))]
            lm_lossA = out.decoder_outputsA.lm_loss
            lm_lossB = out.decoder_outputsB.lm_loss
            lm_lossE = out.decoder_outputsE.lm_loss
            prediction_scoresA = out.decoder_outputsA.lm_logits
            predictionsA = F.log_softmax(prediction_scoresA, dim=2)
            prediction_scoresB = out.decoder_outputsB.lm_logits
            predictionsB = F.log_softmax(prediction_scoresB, dim=2)
            prediction_scoresE = out.decoder_outputsE.lm_logits
            predictionsE = F.log_softmax(prediction_scoresE, dim=2)
            lossa = LLLoss_raw2(predictionsA, alpha_input, ignore_index) 
            lossb = LLLoss_raw2(predictionsB, beta_input, ignore_index) 
            losse = LLLoss_raw2(predictionsE, peptide_input, ignore_index)
            # print(lossa.shape, alpha_input.shape, lossb.shape, beta_input.shape, losse.shape )
            clf_scorea += [lossa[i] for i in range(lossa.shape[0])]
            clf_scoreb += [lossb[i] for i in range(lossb.shape[0])]
            clf_scoree += [losse[i] for i in range(losse.shape[0])]
        # aucab = roc_auc_score(Boolbinders, clf_scoreab)
        # auce = roc_auc_score(Boolbinders, clf_scoree)
        return Boolbinders, clf_scorea, clf_scoreb,clf_scoree



# target_peptide = "NLVPMVATV"
# datasetPetideSpecific= TCRDataset("dataNew/VDJ_final_test.csv", tokenizer, device,target_peptide=target_peptide, mhctok=mhctok)
# # datasetPetideSpecific= TCRDataset("dataNew/VDJ_test_pretune.csv", tokenizer, device,target_peptide=target_peptide, mhctok=mhctok)
# dataloaderPetideSpecific = torch.utils.data.DataLoader(dataset=datasetPetideSpecific, batch_size=512, shuffle=True, collate_fn=datasetValidFinal.all2allmhc_collate_function) 
# Boolbinders, clf_scorea, clf_scoreb,clf_scoree = unsupervised_auc2(model, dataloaderPetideSpecific, tokenizer.pad_token_id)

# def myfunction(losses):
#     return losses[:5].sum()/sum(losses!=0.0)

# len(clf_scorea), len(clf_scoreb), len(clf_scoree)


# out1 = np.array([myfunction(clf_scoree[i]).cpu().item() for i in range(len(clf_scoreb))])
# roc_auc_score(Boolbinders, out1)

# out2 = np.array([myfunction(clf_scorea[i]).cpu().item() for i in range(len(clf_scoreb))])
# roc_auc_score(Boolbinders, out2)

# out3 = np.array([myfunction(clf_scoreb[i]).cpu().item() for i in range(len(clf_scoreb))])
# roc_auc_score(Boolbinders, out3)



# roc_auc_score(Boolbinders, out2/np.sum(out2)  - out1/np.sum(out1))

# out = [myfunction(clf_scorea[i]).cpu().item() +myfunction(clf_scoreb[i]).cpu().item() - myfunction(clf_scoree[i]).cpu().item() for i in range(len(clf_scoreb))]
# roc_auc_score(Boolbinders, out)

# out = [sum(clf_scoreb[i]!=0).cpu().item() for i in range(len(clf_scoreb))]
# roc_auc_score(Boolbinders, out)







#####   categorical MultiTransformer:


# # @add_start_docstrings(ENCODER_DECODER_START_DOCSTRING)
# class Categorical_MultiTransformerModel(PreTrainedModel):
#     r"""
#     [`EncoderDecoderModel`] is a generic model class that will be instantiated as a transformer architecture with one
#     of the base model classes of the library as encoder and another one as decoder when created with the
#     :meth*~transformers.AutoModel.from_pretrained* class method for the encoder and
#     :meth*~transformers.AutoModelForCausalLM.from_pretrained* class method for the decoder.
#     """
#     config_class = EncoderDecoderConfig
#     base_model_prefix = "encoder_decoder"

#     def __init__(
#         self,
#         config: Optional[PretrainedConfig] = None,
#         encoderA: Optional[PreTrainedModel] = None,
#         decoderA: Optional[PreTrainedModel] = None,
#         encoderB: Optional[PreTrainedModel] = None,
#         decoderB: Optional[PreTrainedModel] = None,
#         encoderE: Optional[PreTrainedModel] = None,
#         decoderE: Optional[PreTrainedModel] = None,
#     ):
#         if config is None and (encoderA is None or decoderA is None):
#             raise ValueError("Either a configuration or an encoder and a decoder has to be provided.")
#         if config is None:
#             config = EncoderDecoderConfig.from_encoder_decoder_configs(encoderA.config, decoderA.config)
#         else:
#             if not isinstance(config, self.config_class):
#                 raise ValueError(f"Config: {config} has to be of type {self.config_class}")

#         if config.decoder.cross_attention_hidden_size is not None:
#             if config.decoder.cross_attention_hidden_size != config.encoder.hidden_size:
#                 raise ValueError(
#                     "If `cross_attention_hidden_size` is specified in the decoder's configuration, it has to be equal"
#                     f" to the encoder's `hidden_size`. Got {config.decoder.cross_attention_hidden_size} for"
#                     f" `config.decoder.cross_attention_hidden_size` and {config.encoder.hidden_size} for"
#                     " `config.encoder.hidden_size`."
#                 )

#         # initialize with config
#         super().__init__(config)

#         if encoderA is None:
#             from ..auto.modeling_auto import AutoModel

#             encoderA = AutoModel.from_config(config.encoder)
#         if encoderE is None:
#             from ..auto.modeling_auto import AutoModel

#             encoderE = AutoModel.from_config(config.encoder)
#         if encoderB is None:
#             from ..auto.modeling_auto import AutoModel

#             encoderB = AutoModel.from_config(config.encoder)

#         if decoderA is None:
#             from ..auto.modeling_auto import AutoModelForCausalLM
#             decoderA = AutoModelForCausalLM.from_config(config.decoder)
#         if decoderB is None:
#             from ..auto.modeling_auto import AutoModelForCausalLM
#             decoderB = AutoModelForCausalLM.from_config(config.decoder)
#         if decoderE is None:
#             from ..auto.modeling_auto import AutoModelForCausalLM
#             decoderE = AutoModelForCausalLM.from_config(config.decoder)

#         self.encoderA = encoderA
#         self.decoderA = decoderA
#         self.encoderB = encoderB
#         self.decoderB = decoderB
#         self.encoderE = encoderE
#         self.decoderE = decoderE
#         self.num_labels = 2
#         self.num_epitope = decoderE.config.num_epitope
#         self.MLMHeadA =  BertOnlyMLMHead(decoderA.config)
#         self.MLMHeadB =  BertOnlyMLMHead(decoderB.config)
#         self.MLMHeadE =  BertOnlyMLMHead(decoderE.config)

#         self.mhc_embeddings = nn.Embedding(encoderA.config.mhc_vocab_size, encoderA.config.hidden_size)
#         if self.encoderA.config.to_dict() != self.config.encoder.to_dict():
#             logger.warning(
#                 f"Config of the encoder: {self.encoderA.__class__} is overwritten by shared encoder config:"
#                 f" {self.config.encoder}"
#             )
#         if self.decoderA.config.to_dict() != self.config.decoder.to_dict():
#             logger.warning(
#                 f"Config of the decoder: {self.decoderA.__class__} is overwritten by shared decoder config:"
#                 f" {self.config.decoder}"
#             )

#         # make sure that the individual model's config refers to the shared config
#         # so that the updates to the config will be synced
#         self.encoderA.config = self.config.encoder
#         self.decoderA.config = self.config.decoder
#         self.encoderB.config = self.config.encoder
#         self.decoderB.config = self.config.decoder
#         self.encoderE.config = self.config.encoder
#         # self.decoderE.config = self.config.decoder
#         # encoder outputs might need to be projected to different dimension for decoder
#         if (
#             self.encoderA.config.hidden_size != self.decoderA.config.hidden_size
#             and self.decoderA.config.cross_attention_hidden_size is None
#         ):
#             self.enc_to_dec_proj = nn.Linear(self.encoderA.config.hidden_size, self.decoderA.config.hidden_size)

#         if self.encoderA.get_output_embeddings() is not None:
#             raise ValueError(
#                 f"The encoder {self.encoderA} should not have a LM Head. Please use a model without LM Head"
#             )

#         # tie encoder, decoder weights if config set accordingly
#         self.tie_weights()

#     def tie_weights(self):
#         # tie encoder & decoder if needed
#         if self.config.tie_encoder_decoder:
#             # tie encoder and decoder base model
#             decoder_base_model_prefix = self.decoder.base_model_prefix
#             self._tie_encoder_decoder_weights(
#                 self.encoder, self.decoder._modules[decoder_base_model_prefix], self.decoder.base_model_prefix
#             )

#     def get_encoder(self):
#         return self.encoder

#     def get_decoder(self):
#         return self.decoder

#     def get_input_embeddings(self):
#         return self.encoder.get_input_embeddings()

#     def get_output_embeddings(self):
#         return self.decoder.get_output_embeddings()

#     def set_output_embeddings(self, new_embeddings):
#         return self.decoder.set_output_embeddings(new_embeddings)

#     @classmethod
#     def from_pretrained(cls, *args, **kwargs):
#         # At the moment fast initialization is not supported for composite models
#         if kwargs.get("_fast_init", False):
#             logger.warning(
#                 "Fast initialization is currently not supported for EncoderDecoderModel. "
#                 "Falling back to slow initialization..."
#             )
#         kwargs["_fast_init"] = False
#         return super().from_pretrained(*args, **kwargs)

#     @classmethod
#     def from_encoder_decoder_pretrained(
#         cls,
#         encoder_pretrained_model_name_or_path: str = None,
#         decoder_pretrained_model_name_or_path: str = None,
#         *model_args,
#         **kwargs
#     ) -> PreTrainedModel:
#         r"""
#         Instantiate an encoder and a decoder from one or two base classes of the library from pretrained model
#         checkpoints.
#         The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated). To train
#         the model, you need to first set it back in training mode with `model.train()`.
#         Params:
#             encoder_pretrained_model_name_or_path (`str`, *optional*):
#                 Information necessary to initiate the encoder. Can be either:
#                     - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
#                       Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
#                       user or organization name, like `dbmdz/bert-base-german-cased`.
#                     - A path to a *directory* containing model weights saved using
#                       [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
#                     - A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
#                       this case, `from_tf` should be set to `True` and a configuration object should be provided as
#                       `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
#                       PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
#             decoder_pretrained_model_name_or_path (`str`, *optional*, defaults to `None`):
#                 Information necessary to initiate the decoder. Can be either:
#                     - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
#                       Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
#                       user or organization name, like `dbmdz/bert-base-german-cased`.
#                     - A path to a *directory* containing model weights saved using
#                       [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
#                     - A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
#                       this case, `from_tf` should be set to `True` and a configuration object should be provided as
#                       `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
#                       PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
#             model_args (remaining positional arguments, *optional*):
#                 All remaining positional arguments will be passed to the underlying model's `__init__` method.
#             kwargs (remaining dictionary of keyword arguments, *optional*):
#                 Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
#                 `output_attentions=True`).
#                 - To update the encoder configuration, use the prefix *encoder_* for each configuration parameter.
#                 - To update the decoder configuration, use the prefix *decoder_* for each configuration parameter.
#                 - To update the parent model configuration, do not use a prefix for each configuration parameter.
#                 Behaves differently depending on whether a `config` is provided or automatically loaded.
#         Example:
#         ```python
#         >>> from transformers import EncoderDecoderModel
#         >>> # initialize a bert2bert from two pretrained BERT models. Note that the cross-attention layers will be randomly initialized
#         >>> model = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased", "bert-base-uncased")
#         >>> # saving model after fine-tuning
#         >>> model.save_pretrained("./bert2bert")
#         >>> # load fine-tuned model
#         >>> model = EncoderDecoderModel.from_pretrained("./bert2bert")
#         ```"""

#         kwargs_encoder = {
#             argument[len("encoder_") :]: value for argument, value in kwargs.items() if argument.startswith("encoder_")
#         }

#         kwargs_decoder = {
#             argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
#         }

#         # remove encoder, decoder kwargs from kwargs
#         for key in kwargs_encoder.keys():
#             del kwargs["encoder_" + key]
#         for key in kwargs_decoder.keys():
#             del kwargs["decoder_" + key]

#         # Load and initialize the encoder and decoder
#         # The distinction between encoder and decoder at the model level is made
#         # by the value of the flag `is_decoder` that we need to set correctly.
#         encoder = kwargs_encoder.pop("model", None)
#         if encoder is None:
#             if encoder_pretrained_model_name_or_path is None:
#                 raise ValueError(
#                     "If `encoder_model` is not defined as an argument, a `encoder_pretrained_model_name_or_path` has "
#                     "to be defined."
#                 )

#             if "config" not in kwargs_encoder:
#                 encoder_config, kwargs_encoder = AutoConfig.from_pretrained(
#                     encoder_pretrained_model_name_or_path, **kwargs_encoder, return_unused_kwargs=True
#                 )

#                 if encoder_config.is_decoder is True or encoder_config.add_cross_attention is True:
#                     logger.info(
#                         f"Initializing {encoder_pretrained_model_name_or_path} as a encoder model "
#                         "from a decoder model. Cross-attention and casual mask are disabled."
#                     )
#                     encoder_config.is_decoder = False
#                     encoder_config.add_cross_attention = False

#                 kwargs_encoder["config"] = encoder_config

#             encoder = AutoModel.from_pretrained(encoder_pretrained_model_name_or_path, *model_args, **kwargs_encoder)

#         decoder = kwargs_decoder.pop("model", None)
#         if decoder is None:
#             if decoder_pretrained_model_name_or_path is None:
#                 raise ValueError(
#                     "If `decoder_model` is not defined as an argument, a `decoder_pretrained_model_name_or_path` has "
#                     "to be defined."
#                 )

#             if "config" not in kwargs_decoder:
#                 decoder_config, kwargs_decoder = AutoConfig.from_pretrained(
#                     decoder_pretrained_model_name_or_path, **kwargs_decoder, return_unused_kwargs=True
#                 )

#                 if decoder_config.is_decoder is False or decoder_config.add_cross_attention is False:
#                     logger.info(
#                         f"Initializing {decoder_pretrained_model_name_or_path} as a decoder model. Cross attention"
#                         f" layers are added to {decoder_pretrained_model_name_or_path} and randomly initialized if"
#                         f" {decoder_pretrained_model_name_or_path}'s architecture allows for cross attention layers."
#                     )
#                     decoder_config.is_decoder = True
#                     decoder_config.add_cross_attention = True

#                 kwargs_decoder["config"] = decoder_config

#             if kwargs_decoder["config"].is_decoder is False or kwargs_decoder["config"].add_cross_attention is False:
#                 logger.warning(
#                     f"Decoder model {decoder_pretrained_model_name_or_path} is not initialized as a decoder. "
#                     f"In order to initialize {decoder_pretrained_model_name_or_path} as a decoder, "
#                     "make sure that the attributes `is_decoder` and `add_cross_attention` of `decoder_config` "
#                     "passed to `.from_encoder_decoder_pretrained(...)` are set to `True` or do not pass a "
#                     "`decoder_config` to `.from_encoder_decoder_pretrained(...)`"
#                 )

#             decoder = AutoModelForCausalLM.from_pretrained(decoder_pretrained_model_name_or_path, **kwargs_decoder)

#         # instantiate config with corresponding kwargs
#         config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config, **kwargs)
#         return cls(encoder=encoder, decoder=decoder, config=config)

#     # @add_start_docstrings_to_model_forward(ENCODER_DECODER_INPUTS_DOCSTRING) 
#     # @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
#     def forward(
#         self,
#         input_ids: Optional[torch.LongTensor] = (None, None,None),
#         attention_mask: Optional[torch.FloatTensor] =  (None, None,None),
#         # decoder_input_ids: Optional[torch.LongTensor] =  (None, None,None),
#         # decoder_attention_mask: Optional[torch.BoolTensor] =  (None, None,None),
#         encoder_outputs: Optional[Tuple[torch.FloatTensor]] = (None, None,None),
#         past_key_values: Tuple[Tuple[torch.FloatTensor]] =  (None, None,None),
#         inputs_embeds: Optional[torch.FloatTensor] = (None, None,None),
#         # decoder_inputs_embeds: Optional[torch.FloatTensor] =  (None, None,None),
#         labels: Optional[torch.LongTensor] = None,
#         use_cache: Optional[bool] =  (None, None,None),
#         output_attentions: Optional[bool] =  None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = (None, None,None),
#         mhc=None,
#         **kwargs,
#     ) -> Union[Tuple, Seq2SeqLMOutput]:
#         input_idsA=input_ids[0]
#         input_idsB=input_ids[1]
#         input_idsE=input_ids[2]
        
#         attention_maskA=attention_mask[0]
#         attention_maskB=attention_mask[1]
#         attention_maskE=attention_mask[2]

#         encoder_outputsA=encoder_outputs[0]
#         encoder_outputsB=encoder_outputs[1]
#         encoder_outputsE=encoder_outputs[2]

#         past_key_valuesA=past_key_values[0]
#         past_key_valuesB=past_key_values[1]
#         past_key_valuesE=past_key_values[2]

#         inputs_embedsA=inputs_embeds[0]
#         inputs_embedsB=inputs_embeds[1]
#         inputs_embedsE=inputs_embeds[2]
#         r"""
#         Returns:
#         Examples:
#         ```python
#         >>> from transformers import EncoderDecoderModel, BertTokenizer
#         >>> import torch
#         >>> tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
#         >>> model = EncoderDecoderModel.from_encoder_decoder_pretrained(
#         ...     "bert-base-uncased", "bert-base-uncased"
#         ... )  # initialize Bert2Bert from pre-trained checkpoints
#         >>> # training
#         >>> model.config.decoder_start_token_id = tokenizer.cls_token_id
#         >>> model.config.pad_token_id = tokenizer.pad_token_id
#         >>> model.config.vocab_size = model.config.decoder.vocab_size
#         >>> input_ids = tokenizer("This is a really long text", return_tensors="pt").input_ids
#         >>> labels = tokenizer("This is the corresponding summary", return_tensors="pt").input_ids
#         >>> outputs = model(input_ids=input_ids, labels=input_ids)
#         >>> loss, logits = outputs.loss, outputs.logits
#         >>> # save and load from pretrained
#         >>> model.save_pretrained("bert2bert")
#         >>> model = EncoderDecoderModel.from_pretrained("bert2bert")
#         >>> # generation
#         >>> generated = model.generate(input_ids)
#         ```"""






#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

#         kwargs_decoder = {
#             argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
#         }

#         if encoder_outputsA is None:
#             encoder_outputsA = self.encoderA(
#                 input_ids=input_idsA,
#                 attention_mask=attention_maskA,
#                 inputs_embeds=inputs_embedsA,
#                 output_attentions=output_attentions,
#                 output_hidden_states=output_hidden_states,
#                 return_dict=return_dict,
#                 **kwargs_encoder,
#             )
#         elif isinstance(encoder_outputsA, tuple):
#             encoder_outputsA = BaseModelOutput(*encoder_outputsA)


#         if encoder_outputsB is None:
#             encoder_outputsB = self.encoderB(
#                 input_ids=input_idsB,
#                 attention_mask=attention_maskB,
#                 inputs_embeds=inputs_embedsB,
#                 output_attentions=output_attentions,
#                 output_hidden_states=output_hidden_states,
#                 return_dict=return_dict,
#                 **kwargs_encoder,
#             )
#         elif isinstance(encoder_outputsB, tuple):
#             encoder_outputsB = BaseModelOutput(*encoder_outputsB)

                
#         if encoder_outputsE is None:
#             encoder_outputsE = self.encoderE(
#                 input_ids=input_idsE,
#                 attention_mask=attention_maskE,
#                 inputs_embeds=inputs_embedsE,
#                 output_attentions=output_attentions,
#                 output_hidden_states=output_hidden_states,
#                 return_dict=return_dict,
#                 **kwargs_encoder,
#             )
#         elif isinstance(encoder_outputsE, tuple):
#             encoder_outputsE = BaseModelOutput(*encoder_outputsE)

#         encoder_hidden_statesA = encoder_outputsA[0]
#         encoder_hidden_statesB = encoder_outputsB[0]
#         encoder_hidden_statesE = encoder_outputsE[0]
#         # optionally project encoder_hidden_states
#         if (
#             self.encoderA.config.hidden_size != self.decoderA.config.hidden_size
#             and self.decoderA.config.cross_attention_hidden_size is None
#         ):
#             encoder_hidden_statesA = self.enc_to_dec_proj(encoder_hidden_statesA)

#         if (
#             self.encoderB.config.hidden_size != self.decoderB.config.hidden_size
#             and self.decoderB.config.cross_attention_hidden_size is None
#         ):
#             encoder_hidden_statesB = self.enc_to_dec_proj(encoder_hidden_statesB)

#         if (
#             self.encoderE.config.hidden_size != self.decoderE.config.hidden_size
#             and self.decoderE.config.cross_attention_hidden_size is None
#         ):
#             encoder_hidden_statesE = self.enc_to_dec_proj(encoder_hidden_statesE)

#         # if (labels is not None) and (decoder_input_ids is None and decoder_inputs_embeds is None):
#         #     decoder_input_ids = shift_tokens_right(
#         #         labels, self.config.pad_token_id, self.config.decoder_start_token_id
#         #     )
#         # print(dict(mhc))
#         # print(type(mhc["input_ids"]))
#         #mhc = default_collate(dict(mhc))
#         # print(mhc["input_ids"])
#         # print(torch.tensor(mhc["input_ids"]))
#         mhc_encoded = self.mhc_embeddings(mhc["input_ids"])
#         mhc_attention_mask = mhc["attention_mask"]
#         # Decode
#         labelsA = (labels, input_idsA)
#         decoder_outputsA = self.decoderA(
#             input_ids = input_idsA,
#             attention_mask = attention_maskA,
#             encoder_hidden_states = torch.cat([mhc_encoded,encoder_hidden_statesB, encoder_hidden_statesE], dim=1),
#             encoder_attention_mask = torch.cat([mhc_attention_mask, attention_maskB, attention_maskE], dim=1),
#             inputs_embeds = inputs_embedsA,
#             output_attentions = output_attentions,
#             output_hidden_states = output_hidden_states,
#             labels=labelsA,
#             use_cache=use_cache,
#             past_key_values=past_key_valuesA,
#             return_dict=return_dict,
#             **kwargs_decoder,
#         )
#         pooled_outputA = decoder_outputsA.pooled_output

#         labelsB = (labels, input_idsB)
#         decoder_outputsB = self.decoderB(
#             input_ids = input_idsB,
#             attention_mask = attention_maskB,
#             encoder_hidden_states = torch.cat([mhc_encoded,encoder_hidden_statesA, encoder_hidden_statesE], dim=1),
#             encoder_attention_mask = torch.cat([mhc_attention_mask,attention_maskA, attention_maskE], dim=1),
#             inputs_embeds = inputs_embedsB,
#             output_attentions = output_attentions,
#             output_hidden_states = output_hidden_states,
#             labels=labelsB,
#             use_cache=use_cache,
#             past_key_values=past_key_valuesB,
#             return_dict=return_dict,
#             **kwargs_decoder,
#         )
#         pooled_outputB = decoder_outputsB.pooled_output

#         labelsE = (labels, input_idsE)
#         decoder_outputsE = self.decoderE(
#             input_ids = input_idsE,
#             attention_mask = attention_maskE,
#             encoder_hidden_states = torch.cat([mhc_encoded,encoder_hidden_statesA, encoder_hidden_statesB], dim=1),
#             encoder_attention_mask = torch.cat([mhc_attention_mask,attention_maskA, attention_maskB], dim=1),
#             inputs_embeds = inputs_embedsE,
#             output_attentions = output_attentions,
#             output_hidden_states = output_hidden_states,
#             labels=labelsE,
#             use_cache=use_cache,
#             past_key_values=past_key_valuesE,
#             return_dict=return_dict,
#             **kwargs_decoder,
#         )
#         pooled_outputE= decoder_outputsE.pooled_output

#         pooled_output = torch.cat([pooled_outputA,pooled_outputB,pooled_outputE], dim=1)
#         logits = self.classifier(pooled_output)
#         labelsCLS = labels
#         lossCLS = None
#         if labelsCLS is not None:
#             if self.config.problem_type is None:
#                 if self.num_labels == 1:
#                     self.config.problem_type = "regression"
#                 elif self.num_labels > 1 and (labelsCLS.dtype == torch.long or labelsCLS.dtype == torch.int):
#                     self.config.problem_type = "single_label_classification"
#                 else:
#                     self.config.problem_type = "multi_label_classification"

#             if self.config.problem_type == "regression":
#                 loss_fct = MSELoss()
#                 if self.num_labels == 1:
#                     lossCLS = loss_fct(logits.squeeze(), labelsCLS.squeeze())
#                 else:
#                     lossCLS = loss_fct(logits, labelsCLS)
#             elif self.config.problem_type == "single_label_classification":
#                 if self.reweight == True:
#                     loss_fct = CrossEntropyLoss(reduction="none")
#                     lossCLS = loss_fct(logits.view(-1, self.num_labels), labelsCLS.view(-1))
#                 else:
#                     loss_fct = CrossEntropyLoss()
#                     lossCLS = loss_fct(logits.view(-1, self.num_labels), labelsCLS.view(-1))
#             elif self.config.problem_type == "multi_label_classification":
#                 loss_fct = BCEWithLogitsLoss()
#                 lossCLS = loss_fct(logits, labelsCLS)




#         # Compute loss independent from decoder (as some shift the logits inside them)
#         loss = None
#         # if labels is not None:
#         #     warnings.warn(DEPRECATION_WARNING, FutureWarning)
#         #     logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
#         #     loss_fct = CrossEntropyLoss()
#         #     loss = loss_fct(logits.reshape(-1, self.decoder.config.vocab_size), labels.view(-1))

#         if not return_dict:
#             if loss is not None:
#                 return (loss,) + decoder_outputs + encoder_outputs
#             else:
#                 return decoder_outputs + encoder_outputs

#         return ED_LMOutput(
#             loss = lossCLS,
#             clf_logits=logits,
#             encoder_outputsA = encoder_outputsA,
#             decoder_outputsA = decoder_outputsA,
#             encoder_outputsB = encoder_outputsB,
#             decoder_outputsB = decoder_outputsB,
#             encoder_outputsE = encoder_outputsE,
#             decoder_outputsE = decoder_outputsE,
#         )
        
#         # Seq2SeqLMOutput(
#         #     loss=loss,
#         #     logits=decoder_outputs.logits,
#         #     past_key_values=decoder_outputs.past_key_values,
#         #     decoder_hidden_states=decoder_outputs.hidden_states,
#         #     decoder_attentions=decoder_outputs.attentions,
#         #     cross_attentions=decoder_outputs.cross_attentions,
#         #     encoder_last_hidden_state=encoder_outputs.last_hidden_state,
#         #     encoder_hidden_states=encoder_outputs.hidden_states,
#         #     encoder_attentions=encoder_outputs.attentions,
#         # )

#     def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
#         return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

#     def prepare_inputs_for_generation(
#         self, input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
#     ):
#         decoder_inputs = self.decoder.prepare_inputs_for_generation(input_ids, past=past)
#         decoder_attention_mask = decoder_inputs["attention_mask"] if "attention_mask" in decoder_inputs else None
#         input_dict = {
#             "attention_mask": attention_mask,
#             "decoder_attention_mask": decoder_attention_mask,
#             "decoder_input_ids": decoder_inputs["input_ids"],
#             "encoder_outputs": encoder_outputs,
#             "past_key_values": decoder_inputs["past_key_values"],
#             "use_cache": use_cache,
#         }
#         return input_dict

#     def resize_token_embeddings(self, *args, **kwargs):
#         raise NotImplementedError(
#             "Resizing the embedding layers via the EncoderDecoderModel directly is not supported. Please use the"
#             " respective methods of the wrapped objects (model.encoder.resize_token_embeddings(...) or"
#             " model.decoder.resize_token_embeddings(...))"
#         )

#     def _reorder_cache(self, past, beam_idx):
#         # apply decoder cache reordering here
#         return self.decoder._reorder_cache(past, beam_idx)


#     def set_reweight(self):
#         self.reweight = True



