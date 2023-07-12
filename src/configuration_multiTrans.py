# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class multiTransConfig(PretrainedConfig):
    r"""
    [`multiTransConfig`] is the configuration class to store the configuration of a [`multiTransConfig`]. It is
    used to instantiate an multi Encoder multi Decoder model according to the specified arguments, defining the encoder and decoder
    configs.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        kwargs (*optional*):
            Dictionary of keyword arguments. Notably:

                - **encoder** ([`PretrainedConfig`], *optional*) -- An instance of a configuration object that defines
                  the encoder config.
                - **decoder** ([`PretrainedConfig`], *optional*) -- An instance of a configuration object that defines
                  the decoder config.

    Examples:

    ```python
    >>> from transformers import BertConfig, EncoderDecoderConfig, EncoderDecoderModel

    >>> # Initializing a BERT bert-base-uncased style configuration
    >>> config_encoder = BertConfig()
    >>> config_decoder = BertConfig()

    >>> config = EncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)

    >>> # Initializing a Bert2Bert model from the bert-base-uncased style configurations
    >>> model = EncoderDecoderModel(config=config)

    >>> # Accessing the model configuration
    >>> config_encoder = model.config.encoder
    >>> config_decoder = model.config.decoder
    >>> # set decoder config to causal lm
    >>> config_decoder.is_decoder = True
    >>> config_decoder.add_cross_attention = True

    >>> # Saving the model, including its configuration
    >>> model.save_pretrained("my-model")

    >>> # loading model and config from pretrained folder
    >>> encoder_decoder_config = EncoderDecoderConfig.from_pretrained("my-model")
    >>> model = EncoderDecoderModel.from_pretrained("my-model", config=encoder_decoder_config)
    ```"""
    model_type = "encoder-decoder"
    is_composition = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert (
            "encoder" in kwargs and "decoder" in kwargs
        ), "Config has to be initialized with encoder and decoder config"
        encoderA_config = kwargs.pop("encoderA")
        encoderA_model_type = encoderA_config.pop("model_typeA")
        decoderA_config = kwargs.pop("decoderA")
        decoderA_model_type = decoderA_config.pop("model_type")

        encoderB_config = kwargs.pop("encoder")
        encoderB_model_type = encoderA_config.pop("model_type")
        decoderB_config = kwargs.pop("decoder")
        decoderB_model_type = decoderA_config.pop("model_type")

        encoderE_config = kwargs.pop("encoder")
        encoderE_model_type = encoderE_config.pop("model_type")
        decoderE_config = kwargs.pop("decoder")
        decoderE_model_type = decoderE_config.pop("model_type")

        from transformers.models.auto.configuration_auto import AutoConfig

        self.encoderA = AutoConfig.for_model(encoderA_model_type, **encoderA_config)
        self.decoderA = AutoConfig.for_model(decoderA_model_type, **decoderA_config)

        self.encoderB = AutoConfig.for_model(encoderB_model_type, **encoderB_config)
        self.decoderB = AutoConfig.for_model(decoderB_model_type, **decoderB_config)

        self.encoderE = AutoConfig.for_model(encoderE_model_type, **encoderE_config)
        self.decoderE = AutoConfig.for_model(decoderE_model_type, **decoderE_config)
        self.is_encoder_decoder = True

    @classmethod
    def from_encoder_decoder_configs(
        cls, encoderA_config: PretrainedConfig, decoderA_config: PretrainedConfig,
        encoderB_config: PretrainedConfig, decoderB_config: PretrainedConfig,
        encoderE_config: PretrainedConfig, decoderE_config: PretrainedConfig,
         **kwargs
    ) -> PretrainedConfig:
        r"""
        Instantiate a [`EncoderDecoderConfig`] (or a derived class) from a pre-trained encoder model configuration and
        decoder model configuration.

        Returns:
            [`EncoderDecoderConfig`]: An instance of a configuration object
        """
        logger.info("Set `config.is_decoder=True` and `config.add_cross_attention=True` for decoder_config")
        decoderA_config.is_decoder = True
        decoderA_config.add_cross_attention = True
        decoderB_config.is_decoder = True
        decoderB_config.add_cross_attention = True
        decoderE_config.is_decoder = True
        decoderE_config.add_cross_attention = True

        return cls(encoderA=encoderA_config.to_dict(), decoderA=decoderA_config.to_dict(), 
                encoderB=encoderB_config.to_dict(), decoderB=decoderB_config.to_dict(), 
                encoderE=encoderE_config.to_dict(), decoderE=decoderE_config.to_dict(), 
                **kwargs)

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default *to_dict()* from *PretrainedConfig*.

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["encoder"] = self.encoder.to_dict()
        output["decoder"] = self.decoder.to_dict()
        output["model_type"] = self.__class__.model_type
        return output
