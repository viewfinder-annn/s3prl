from collections import OrderedDict
from typing import Dict, List, Union

import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

SAMPLE_RATE = 16000
HIDDEN_DIM = 1024

class UpstreamExpert(nn.Module):
    def __init__(self, ckpt: str = None, model_config: str = None, **kwargs):
        """
        Args:
            ckpt:
                The checkpoint path for loading your pretrained weights.
                Can be assigned by the -k option in run_downstream.py

            model_config:
                The config path for constructing your model.
                Might not needed if you also save that in your checkpoint file.
                Can be assigned by the -g option in run_downstream.py
        """
        super().__init__()
        self.name = "[Example UpstreamExpert]"

        print(
            f"{self.name} - You can use model_config to construct your customized model: {model_config}"
        )
        print(f"{self.name} - You can use ckpt to load your pretrained weights: {ckpt}")
        print(
            f"{self.name} - If you store the pretrained weights and model config in a single file, "
            "you can just choose one argument (ckpt or model_config) to pass. It's up to you!"
        )

        # The model needs to be a nn.Module for finetuning, not required for representation extraction
        self.model1 = nn.Linear(1, HIDDEN_DIM)
        self.model2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)

    def get_downsample_rates(self, key: str) -> int:
        """
        Since we do not do any downsampling in this example upstream
        All keys' corresponding representations have downsample rate of 1
        """
        return 1

    def forward(self, wavs: List[Tensor]) -> Dict[str, Union[Tensor, List[Tensor]]]:
        """
        When the returning Dict contains the List with more than one Tensor,
        those Tensors should be in the same shape to train a weighted-sum on them.
        """

        wavs = pad_sequence(wavs, batch_first=True).unsqueeze(-1)
        # wavs: (batch_size, max_len, 1)

        hidden = self.model1(wavs)
        # hidden: (batch_size, max_len, hidden_dim)

        feature = self.model2(hidden)
        # feature: (batch_size, max_len, hidden_dim)

        # The "hidden_states" key will be used as default in many cases
        # Others keys in this example are presented for SUPERB Challenge
        return {
            "hidden_states": [hidden, feature],
            "PR": [hidden, feature],
            "ASR": [hidden, feature],
            "QbE": [hidden, feature],
            "SID": [hidden, feature],
            "ASV": [hidden, feature],
            "SD": [hidden, feature],
            "ER": [hidden, feature],
            "SF": [hidden, feature],
            "SE": [hidden, feature],
            "SS": [hidden, feature],
            "secret": [hidden, feature],
        }

import os
import sys
sys.path.append("/mnt/workspace/home/zhangjunan/YuE/inference/xcodec_mini_infer")
sys.path.append("/mnt/workspace/home/zhangjunan/YuE/inference/xcodec_mini_infer/descriptaudiocodec")
import torch
import torchaudio
from torchaudio.transforms import Resample
import numpy as np
from omegaconf import OmegaConf

from models.soundstream_hubert_new import SoundStream

class XCodecUpstreamExpert(nn.Module):
    def __init__(self,
                #  ckpt: str = ,
                #  model_config: str = ,
                 **kwargs):
        super().__init__()
        self.name = "[XCodec UpstreamExpert]"
        # print(f"ckpt: {ckpt}")
        # print(f"model_config: {model_config}")
        config = OmegaConf.load("/mnt/workspace/home/zhangjunan/YuE/inference/xcodec_mini_infer/final_ckpt/config.yaml")
        self.codec_model = eval(config.generator.name)(**config.generator.config)
        params = torch.load("/mnt/workspace/home/zhangjunan/YuE/inference/xcodec_mini_infer/final_ckpt/ckpt_00360000.pth")
        self.codec_model.load_state_dict(params['codec_model'])
        self.codec_model.eval()

    def get_downsample_rates(self, key: str) -> int:
        """
        Since we do not do any downsampling in this example upstream
        All keys' corresponding representations have downsample rate of 1
        """
        return 320

    def forward(self, wavs: List[Tensor]) -> Dict[str, Union[Tensor, List[Tensor]]]:
        """
        When the returning Dict contains the List with more than one Tensor,
        those Tensors should be in the same shape to train a weighted-sum on them.
        """

        wavs = pad_sequence(wavs, batch_first=True).unsqueeze(-1).transpose(1, 2) # (batch_size, 1, max_len)

        codes = self.codec_model.encode(wavs, target_bw=4)
        embed = self.codec_model.get_embed(codes).transpose(1, 2) # (batch_size, max_len // 320, hidden_dim)

        # The "hidden_states" key will be used as default in many cases
        # Others keys in this example are presented for SUPERB Challenge
        return {
            "hidden_states": [embed],
            "PR": [embed],
            "ASR": [embed],
            "QbE": [embed],
            "SID": [embed],
            "ASV": [embed],
            "SD": [embed],
            "ER": [embed],
            "SF": [embed],
            "SE": [embed],
            "SS": [embed],
            "secret": [embed],
        }

