# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import os
from typing import Union

import numpy as np
import pandas as pd
import torch
from omegaconf import ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

import verl.utils.torch_functional as verl_F
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.model import compute_position_id_with_mask


class SFTDataset(Dataset):
    """Simple supervised dataset for Search-R1 style cold-start traces.

    The parquet files must contain a prompt column and a response column. The
    prompt may be either a raw string or a chat list such as
    [{"role": "user", "content": "..."}]. The response is trained with a
    causal-LM loss, while the prompt tokens are masked out.
    """

    def __init__(
        self,
        parquet_files: Union[str, list[str]],
        tokenizer: PreTrainedTokenizer,
        prompt_key: str = "question",
        prompt_dict_keys: list[str] | None = None,
        response_key: str = "answer",
        response_dict_keys: list[str] | None = None,
        max_length: int = 1024,
        truncation: str = "error",
        cache_dir: str = "~/.cache/verl/sft",
    ):
        if not isinstance(parquet_files, (list, ListConfig)):
            parquet_files = [parquet_files]

        self.parquet_files = list(parquet_files)
        self.cache_dir = os.path.expanduser(cache_dir)
        self.tokenizer = tokenizer
        self.prompt_key = prompt_key
        self.prompt_dict_keys = prompt_dict_keys
        self.response_key = response_key
        self.response_dict_keys = response_dict_keys
        self.max_length = max_length
        self.truncation = truncation

        self._download()
        self._read_files()

    def _download(self) -> None:
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_local_path_from_hdfs(src=parquet_file, cache_dir=self.cache_dir)

    def _read_files(self) -> None:
        dataframes = [pd.read_parquet(path) for path in self.parquet_files]
        self.dataframe = pd.concat(dataframes, ignore_index=True)
        print(f"SFT dataset len: {len(self.dataframe)}")

    def __len__(self) -> int:
        return len(self.dataframe)

    def _build_prompt_text(self, value) -> str:
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, list):
            if self.tokenizer.chat_template:
                return self.tokenizer.apply_chat_template(value, add_generation_prompt=True, tokenize=False)
            return "\n".join(str(item.get("content", item)) if isinstance(item, dict) else str(item) for item in value)
        return str(value)

    def _build_response_text(self, value) -> str:
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, list):
            return "\n".join(str(item.get("content", item)) if isinstance(item, dict) else str(item) for item in value)
        return str(value)

    def __getitem__(self, item: int) -> dict:
        row = self.dataframe.iloc[item].to_dict()
        prompt_text = self._build_prompt_text(row[self.prompt_key])
        response_text = self._build_response_text(row[self.response_key])
        if self.tokenizer.eos_token and not response_text.endswith(self.tokenizer.eos_token):
            response_text = response_text + self.tokenizer.eos_token

        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        response_ids = self.tokenizer(response_text, add_special_tokens=False)["input_ids"]
        input_ids = prompt_ids + response_ids
        loss_mask = [0] * len(prompt_ids) + [1] * len(response_ids)

        if len(input_ids) > self.max_length:
            if self.truncation == "error":
                raise ValueError(f"SFT sample length {len(input_ids)} exceeds max_length={self.max_length}")
            if self.truncation == "left":
                input_ids = input_ids[-self.max_length :]
                loss_mask = loss_mask[-self.max_length :]
            elif self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                loss_mask = loss_mask[: self.max_length]
            else:
                raise ValueError(f"Unsupported truncation mode: {self.truncation}")

        input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
        loss_mask_tensor = torch.tensor(loss_mask, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.ones_like(input_tensor)

        pad_id = self.tokenizer.pad_token_id
        input_tensor = verl_F.pad_sequence_to_length(input_tensor, self.max_length, pad_id, left_pad=False)
        attention_mask = verl_F.pad_sequence_to_length(attention_mask, self.max_length, 0, left_pad=False)
        loss_mask_tensor = verl_F.pad_sequence_to_length(loss_mask_tensor, self.max_length, 0, left_pad=False)
        position_ids = compute_position_id_with_mask(attention_mask)

        return {
            "input_ids": input_tensor[0],
            "attention_mask": attention_mask[0],
            "position_ids": position_ids[0],
            "loss_mask": loss_mask_tensor[0],
        }
