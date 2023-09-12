"""
This code is mostly a modification of the original code from the hugginface library:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/encoder_decoder/modeling_encoder_decoder.py
"""

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.utils.data as data
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from typing import List, Optional, Tuple, Union, Any, Dict
from torch.utils.data._utils.collate import default_collate
from sklearn.metrics import roc_auc_score

from transformers import BertModel, PretrainedConfig, AutoModelForCausalLM
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertOnlyMLMHead, SequenceClassifierOutput
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput, ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import  logging
from transformers.models.encoder_decoder.configuration_encoder_decoder import EncoderDecoderConfig
import warnings

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

class TCRDataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader. for TCR data."""
    def __init__(self, csv_file, tokenizer, device, target_binder=None, target_peptide=None, excluded_peptide=None, mhctok=None):#, alpha_maxlength, beta_maxlength, epitope_maxlength):
        self.device=device
        self.tokenizer = tokenizer
        print("Loading the data ...")
        df = pd.read_csv(csv_file)
        
        if target_binder:
            df = df[df["binder"]==1]

        if target_peptide:
            df = df[df["peptide"].apply(lambda x: x in target_peptide)]

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

    def __getitem__(self, offset):
        """Return one datapoint from the dataset, at position offset in the table.
            - if reweight is True, will provide a weight for each datapoint.
            - if mhctok is provided will provide an mhc token for each datapoint.
        """
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
        """Set the weights for each datapoint, based on the frequency of the peptide in the dataset."""
        freq = self.df["peptide"].value_counts()/self.df["peptide"].value_counts().sum()
        alpha = alpha
        freq = alpha*freq + (1-alpha)/len(self.df["peptide"].value_counts())
        self.weights = (1/torch.tensor(list(self.df.apply(lambda x: freq[x["peptide"]],1 ))))/len(self.df["peptide"].value_counts())
        self.reweight = True

    def all2allmhc_collate_function(self, batch):
        """Collate function for the Tulip model returning peptide, alpha, beta, binder, mhc and weight if reweight is True"""

        if self.reweight:
            (alpha, beta, peptide, binder, mhc, weight) = zip(*batch)
        else:
            (alpha, beta, peptide, binder, mhc) = zip(*batch)

        peptide = self.tokenizer(list(peptide),padding="longest", add_special_tokens=True)
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
        ele = torch.arange(0, hidden_states.shape[0])

        first_token_tensor = hidden_states[ele.long(), targetind.long()]#.gather(1, targetind.view(-1,1))#hidden_states[:, -1]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output



class TulipPetal(BertPreTrainedModel):
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
        return_dict = True# return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
          use_cache = False

        # get clfPosition:
        temp = input_ids != self.pad_token_id
        targetind  = torch.sum(temp, dim=1) - 1



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
    

    def prepare_inputs_for_generation(self, input_ids, past=None, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past}
    






class ED_LMOutput(ModelOutput):
    clf_loss: Optional[torch.FloatTensor] = None
    clf_logits: Optional[torch.FloatTensor] = None
    decoder_outputsA = None
    encoder_outputsA = None
    decoder_outputsB = None
    encoder_outputsB = None
    decoder_outputsE = None
    encoder_outputsE = None


logger = logging.get_logger(__name__)


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



class Tulip(PreTrainedModel):
    config_class = EncoderDecoderConfig
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
        self.reweight=False
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
        # This classifier is only here for potential future supervised task
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

        if (
            self.encoderA.config.hidden_size != self.decoderA.config.hidden_size
            and self.decoderA.config.cross_attention_hidden_size is None
        ):
            self.enc_to_dec_proj = nn.Linear(self.encoderA.config.hidden_size, self.decoderA.config.hidden_size)

        if self.encoderA.get_output_embeddings() is not None:
            raise ValueError(
                f"The encoder {self.encoderA} should not have a LM Head. Please use a model without LM Head"
            )

        self.tie_weights()

    def tie_weights(self):
        # tie encoder & decoder if needed
        if self.config.tie_encoder_decoder:
            # tie encoder and decoder base model
            decoder_base_model_prefix = self.decoder.base_model_prefix
            self._tie_encoder_decoder_weights(
                self.encoder, self.decoder._modules[decoder_base_model_prefix], self.decoder.base_model_prefix
            )

    def get_encoder(self, encoder_name='B'):
        if encoder_name=='A':
            return self.encoderA
        elif encoder_name=='B':
            return self.encoderB
        elif encoder_name=='E':
            return self.encoderE


    def get_decoder(self, decoder_name='B'):
        if decoder_name=='A':
            return self.decoderA
        elif decoder_name=='B':
            return self.decoderB
        elif decoder_name=='E':
            return self.decoderE



    def get_input_embeddings(self, encoder_name='B'):
        if encoder_name=='A':
            return self.encoderA.get_input_embeddings()
        elif encoder_name=='B':
            return self.encoderB.get_input_embeddings()
        elif encoder_name=='E':
            return self.encoderE.get_input_embeddings()
        


    def get_output_embeddings(self, decoder_name='B'):
        if decoder_name=='A':
            return self.decoderA.get_output_embeddings()
        elif decoder_name=='B':
            return self.decoderB.get_output_embeddings()
        elif decoder_name=='E':
            return self.decoderE.get_output_embeddings()
        


    def set_output_embeddings(self, new_embeddings, decoder_name='B'):
        if decoder_name=='A':
            return self.decoderA.set_output_embeddings(new_embeddings)
        elif decoder_name=='B':
            return self.decoderB.set_output_embeddings(new_embeddings)
        elif decoder_name=='E':
            return self.decoderE.set_output_embeddings(new_embeddings)
        


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
        togenerate=None,
        **kwargs,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        print('forward', input_ids)
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

        mhc_encoded = self.mhc_embeddings(mhc["input_ids"])
        mhc_attention_mask = mhc["attention_mask"]
        # Decode
        if togenerate not in ['B','E']:
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

        if togenerate not in ['A','E']:
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

        if togenerate not in ['A','B']:
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

        # pooled_output = torch.cat([pooled_outputA,pooled_outputB,pooled_outputE], dim=1)
        # logits = self.classifier(pooled_output)
        labelsCLS = labels
        lossCLS = None
        logits = None
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

        if togenerate == 'A':
            return Seq2SeqLMOutput(
                loss=loss,
                logits=decoder_outputsA.lm_logits,
                past_key_values=decoder_outputsA.past_key_values,
                decoder_hidden_states=decoder_outputsA.hidden_states,
                decoder_attentions=decoder_outputsA.attentions,
                cross_attentions=decoder_outputsA.cross_attentions,
                encoder_last_hidden_state=torch.cat([mhc_encoded,encoder_hidden_statesB, encoder_hidden_statesE], dim=1),
                encoder_hidden_states=encoder_outputsE.hidden_states,
                encoder_attentions=torch.cat([mhc_attention_mask, attention_maskB, attention_maskE], dim=1),
            )
        elif togenerate == 'B':
            return Seq2SeqLMOutput(
                loss=loss,
                logits=decoder_outputsB.lm_logits,
                past_key_values=decoder_outputsB.past_key_values,
                decoder_hidden_states=decoder_outputsB.hidden_states,
                decoder_attentions=decoder_outputsB.attentions,
                cross_attentions=decoder_outputsB.cross_attentions,
                encoder_last_hidden_state=torch.cat([mhc_encoded,encoder_hidden_statesA, encoder_hidden_statesE], dim=1),
                encoder_hidden_states=encoder_outputsE.hidden_states,
                encoder_attentions=torch.cat([mhc_attention_mask,attention_maskA, attention_maskE], dim=1),
            )
        elif togenerate == 'E':
            return Seq2SeqLMOutput(
                loss=loss,
                logits=decoder_outputsE.lm_logits,
                past_key_values=decoder_outputsE.past_key_values,
                decoder_hidden_states=decoder_outputsE.hidden_states,
                decoder_attentions=decoder_outputsE.attentions,
                cross_attentions=decoder_outputsE.cross_attentions,
                encoder_last_hidden_state=torch.cat([mhc_encoded,encoder_hidden_statesA, encoder_hidden_statesB], dim=1),
                encoder_hidden_states=encoder_outputsE.hidden_states,
                encoder_attentions=torch.cat([mhc_attention_mask,attention_maskA, attention_maskB], dim=1),
            )
        

        # if not return_dict:
        #     if loss is not None:
        #         return (loss,) + decoder_outputs + encoder_outputs
        #     else:
        #         return decoder_outputs + encoder_outputs
        else:
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


    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        print('prepare_decoder_input_ids_from_labels')
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    def prepare_inputs_for_generation(
        self, input_ids, past=None, attention_mask=(None, None, None), use_cache=None, encoder_outputs=(None, None, None), **kwargs
    ):
        print('prepare_inputs_for_generation')
        togenerate = kwargs['togenerate']
        if togenerate == 'A':
            decoder_inputs = self.decoderA.prepare_inputs_for_generation(input_ids, past=past)
            decoder_attention_mask = decoder_inputs["attention_mask"] if "attention_mask" in decoder_inputs else None
            input_dict = {
                "input_ids": (decoder_inputs["input_ids"], None, None),
                "attention_mask": attention_mask,
                "decoder_attention_mask": (decoder_attention_mask,None,None),
                "decoder_input_ids": decoder_inputs["input_ids"],
                "encoder_outputs": encoder_outputs,
                "past_key_values": (decoder_inputs["past_key_values"], None, None),
                "use_cache": use_cache,
                "mhc": kwargs['mhc'],
            }
            return input_dict
        elif togenerate == 'B':
            decoder_inputs = self.decoderB.prepare_inputs_for_generation(input_ids, past=past)
            decoder_attention_mask = decoder_inputs["attention_mask"] if "attention_mask" in decoder_inputs else None
            input_dict = {
                "input_ids": (None,decoder_inputs["input_ids"], None),
                "attention_mask": attention_mask,
                "decoder_attention_mask": (None,decoder_attention_mask,None),
                "decoder_input_ids": decoder_inputs["input_ids"],
                "encoder_outputs": encoder_outputs,
                "past_key_values": (None, decoder_inputs["past_key_values"], None),
                "use_cache": use_cache,
                "mhc": kwargs['mhc'],
            }
            return input_dict
        elif togenerate == 'E':
            decoder_inputs = self.decoderE.prepare_inputs_for_generation(input_ids, past=past)
            decoder_attention_mask = decoder_inputs["attention_mask"] if "attention_mask" in decoder_inputs else None
            input_dict = {
                "input_ids": (None,None,decoder_inputs["input_ids"]),
                "attention_mask": attention_mask,
                "decoder_attention_mask": (None,None,decoder_attention_mask),
                "decoder_input_ids": decoder_inputs["input_ids"],
                "encoder_outputs": encoder_outputs,
                "past_key_values": (None, None,  decoder_inputs["past_key_values"]),
                "use_cache": use_cache,
                "mhc": kwargs['mhc'],
            }
            return input_dict
        else:
            raise ValueError('togenerate should be A, B or E')

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

    
    def _prepare_model_inputs(
        self,
        inputs: Optional[torch.Tensor] = None,
        bos_token_id: Optional[int] = None,
        model_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[str], Dict[str, torch.Tensor]]:
        """
        This function extracts the model-specific `inputs` for generation.
        """
        print('_prepare_model_inputs')

        # 1. retrieve all kwargs that are non-None or non-model input related.
        # some encoder-decoder models have different names for model and encoder
        # if (
        #     self.config.is_encoder_decoder
        #     and hasattr(self, "encoder")
        #     and self.encoder.main_input_name != self.main_input_name
        # ):
        #     input_name = self.encoder.main_input_name
        # else:
        #     input_name = self.main_input_name


        input_name = "input_ids"
        model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None or k != input_name}

        # 2. check whether model_input_name is passed as kwarg
        # if yes and `inputs` is None use kwarg inputs
        # inputs_kwarg = model_kwargs.pop(input_name, None)
        # if inputs_kwarg is not None and inputs is not None:
        #     raise ValueError(
        #         f"`inputs`: {inputs}` were passed alongside "
        #         f"{input_name} which is not allowed."
        #         f"Make sure to either pass {inputs} or {input_name}=..."
        #     )
        # elif inputs_kwarg is not None:
        #     inputs = inputs_kwarg

        # # 3. models with `input_ids` can also make use of `inputs_embeds`
        # if self._can_retrieve_inputs_from_name(inputs, "inputs_embeds", model_kwargs):
        #     inputs, input_name = model_kwargs["inputs_embeds"], "inputs_embeds"

        # 4. Only encoder-decoder models can have non `input_ids` input format
        # if not self.config.is_encoder_decoder and input_name != "input_ids":
        #     raise ValueError(
        #         f"If {input_name} is passed as model-specific keyword "
        #         "input then model has to be an encoder-decoder and not a "
        #         f"{self.__class__.__name__}."
        #     )

        # 5. if `inputs` is still None, try to create `input_ids` from BOS token
        if inputs is None:
            inputs = torch.ones((1,1), dtype=torch.long, device=device) * bos_token_id
            # self._prepare_input_ids_for_generation(bos_token_id, model_kwargs.get("encoder_outputs"))

        return inputs, input_name, model_kwargs

    def _prepare_encoder_decoder_kwargs_for_generation(
        self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None
    ) -> Dict[str, Any]:
        # 1. get encoder
        # encoder = self.get_encoder()

        # 2. prepare encoder args and encoder kwargs from model kwargs
        # irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        # encoder_kwargs = {
        #     argument: value
        #     for argument, value in model_kwargs.items()
        #     if not any(argument.startswith(p) for p in irrelevant_prefix)
        # }

        print('_prepare_encoder_decoder_kwargs_for_generation', inputs_tensor)
        encoder_kwargs = model_kwargs.copy()
        encoder_kwargs["togenerate"] = None

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs["return_dict"] = True
        # encoder_kwargs[model_input_name] = inputs_tensor
        # model_kwargs["encoder_outputs"]: ModelOutput 
        out = self.forward(**encoder_kwargs)
        model_kwargs["encoder_outputs"] = (out.encoder_outputsA, out.encoder_outputsB, out.encoder_outputsE)
        model_kwargs["decoder_input_ids"] = inputs_tensor
        model_kwargs.pop("input_ids", None) #### WHY?
        return model_kwargs


    @staticmethod
    def _update_model_kwargs_for_generation(
        outputs: ModelOutput, model_kwargs: Dict[str, Any], is_encoder_decoder: bool = False
    ) -> Dict[str, Any]:
        print('_update_model_kwargs_for_generation')
        # update past
        if "past_key_values" in outputs:
            model_kwargs["past"] = outputs.past_key_values
        elif "mems" in outputs:
            model_kwargs["past"] = outputs.mems
        elif "past_buckets_states" in outputs:
            model_kwargs["past"] = outputs.past_buckets_states
        else:
            model_kwargs["past"] = None

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        # update attention mask
        if not is_encoder_decoder:
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )

        return model_kwargs
    

    def _prepare_decoder_input_ids_for_generation(
        self,
        batch_size: int,
        decoder_start_token_id: int = None,
        bos_token_id: int = None,
        model_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        device: torch.device = None,
    ) -> torch.LongTensor:
        print('_prepare_decoder_input_ids_for_generation')
        if model_kwargs is not None and "decoder_input_ids" in model_kwargs:
            return model_kwargs.pop("decoder_input_ids")
        else:
            decoder_start_token_id = self._get_decoder_start_token_id(decoder_start_token_id, bos_token_id)
            if device is None:
                device = self.device
            return torch.ones((batch_size, 1), dtype=torch.long, device=device) * decoder_start_token_id


    @staticmethod
    def _expand_inputs_for_generation(
        input_ids: torch.LongTensor,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        attention_mask: Optional[torch.LongTensor] = (None,None,None),
        encoder_outputs: Optional[Tuple[ModelOutput]] = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        print('_expand_inputs_for_generation')
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)
        model_kwargs["mhc"]["input_ids"] = model_kwargs["mhc"]["input_ids"].index_select(0, expanded_return_idx)
        model_kwargs["mhc"]["attention_mask"] = model_kwargs["mhc"]["attention_mask"].index_select(0, expanded_return_idx)
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = token_type_ids.index_select(0, expanded_return_idx)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = (attention_mask[0].index_select(0, expanded_return_idx),
                                                attention_mask[1].index_select(0, expanded_return_idx),
                                                attention_mask[2].index_select(0, expanded_return_idx))
        

        if is_encoder_decoder:
            if encoder_outputs == (None,None,None):
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            encoder_outputs[0]["last_hidden_state"] = encoder_outputs[0].last_hidden_state.index_select(
                0, expanded_return_idx.to(encoder_outputs[0].last_hidden_state.device)
            )
            encoder_outputs[1]["last_hidden_state"] = encoder_outputs[1].last_hidden_state.index_select(
                0, expanded_return_idx.to(encoder_outputs[1].last_hidden_state.device)
            )
            encoder_outputs[2]["last_hidden_state"] = encoder_outputs[2].last_hidden_state.index_select(
                0, expanded_return_idx.to(encoder_outputs[2].last_hidden_state.device)
            )
            model_kwargs["encoder_outputs"] = encoder_outputs
        return input_ids, model_kwargs

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
        

        alpha_observed_mask = alpha_input.clone().detach()[:,1] != 4
        beta_observed_mask = beta_input.clone().detach()[:,1] != 4
        peptide_observed_mask = peptide_input.clone().detach()[:,1] != 4
        clf_label = binder.clone()
        labels = clf_label

        out = model(input_ids=(alpha_input,beta_input,peptide_input), attention_mask=(alpha_mask,beta_mask,peptide_mask),
                                        labels=labels, mhc=mhc)

        prediction_scoresA = out.decoder_outputsA.lm_logits
        predictionsA = F.log_softmax(prediction_scoresA, dim=2)
        prediction_scoresB = out.decoder_outputsB.lm_logits
        predictionsB = F.log_softmax(prediction_scoresB, dim=2)
        prediction_scoresE = out.decoder_outputsE.lm_logits
        predictionsE = F.log_softmax(prediction_scoresE, dim=2)
        lossa = compute_loss(predictionsA[alpha_observed_mask], alpha_input[alpha_observed_mask], criterion) 
        lossb = compute_loss(predictionsB[beta_observed_mask], beta_input[beta_observed_mask],criterion) 
        losse = compute_loss(predictionsE[peptide_observed_mask], peptide_input[peptide_observed_mask], criterion)

        mlm_lossA = MLM_Loss(model.encoderA, model.MLMHeadA, masker, alpha_input, alpha_mask, alpha_observed_mask)
        mlm_lossB = MLM_Loss(model.encoderB, model.MLMHeadB, masker, beta_input, beta_mask, beta_observed_mask)
        mlm_lossE = MLM_Loss(model.encoderE, model.MLMHeadE, masker, peptide_input, peptide_mask, peptide_observed_mask)

        loss = mlm_lossA+mlm_lossB+mlm_lossE+alph*(lossa+lossb+losse)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
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








def eval_unsupervised(model, masker, test_dataloader, criterion):
    model.eval()
    with torch.no_grad():

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
            
            alpha_observed_mask = torch.tensor(alpha_input)[:,1] != 4
            beta_observed_mask = torch.tensor(beta_input)[:,1] != 4
            peptide_observed_mask = torch.tensor(peptide_input)[:,1] != 4
            # lm_labels = alphabeta_output.clone()
            clf_label = binder.clone()
            labels = clf_label

            out = model(input_ids=(alpha_input,beta_input,peptide_input), attention_mask=(alpha_mask,beta_mask,peptide_mask),
                                            labels=labels, mhc=mhc)


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
            clf_scorea += [-1*lossa[i].cpu().item() for i in range(len(lossa))]
            clf_scoreb += [-1*lossb[i].cpu().item() for i in range(len(lossb))]
            clf_scoree += [-1*losse[i].cpu().item() for i in range(len(losse))]

        auca = roc_auc_score(Boolbinders, clf_scorea)
        aucb = roc_auc_score(Boolbinders, clf_scoreb)
        auce = roc_auc_score(Boolbinders, clf_scoree)
        return auca, aucb, auce







def get_logscore(dataset, model, ignore_index):
    dataloaderPetideSpecific = torch.utils.data.DataLoader(dataset=dataset, batch_size=100, shuffle=False, collate_fn=dataset.all2allmhc_collate_function)
    model.eval()
    with torch.no_grad():


        clf_scoree = []
        Boolbinders = []
        for i, (peptide, alpha, beta, binder, mhc) in enumerate(dataloaderPetideSpecific):
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

            clf_label = binder.clone()
            labels = clf_label

            out = model(input_ids=(alpha_input,beta_input,peptide_input), attention_mask=(alpha_mask,beta_mask,peptide_mask),
                                            labels=labels, mhc=mhc)

            def softm(x):
                x = torch.exp(x)
                su = torch.sum(x, dim=1)
                x = x/su
                return x

            prediction_scoresE = out.decoder_outputsE.lm_logits
            predictionsE = F.log_softmax(prediction_scoresE, dim=2)

            losse = LLLoss_raw(predictionsE, peptide_input, ignore_index)
            clf_scoree += [losse[i].cpu().item() for i in range(len(losse))]
        return clf_scoree


