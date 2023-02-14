import json
import torch
import os
import numpy as np

import random
from transformers import AutoTokenizer
from tools import shift_tokens_right

class DRFormatter:
    def __init__(self, config, mode, *args, **params):
        self.mode = mode
        self.plmpath = os.path.join(config.get("model", "pretrained_model_path"), config.get("model", "pretrained_model"))
        self.tokenizer = AutoTokenizer.from_pretrained(self.plmpath)
        self.query_len = config.getint("train", "query_len")
        self.ctx_len = config.getint("train", "ctx_len")
        self.num_train_passage = config.getint("train", "num_train_passage")

    def process(self, data):
        queries = [doc["query"] for doc in data]
        ctxs = []
        for doc in data:
            positive = random.choice(doc["positive_passages"])["text"]
            negative = [d["text"] for d in random.sample(doc["negative_passages"], self.num_train_passage - 1)]
            ctxs.extend([positive] + negative)

        query_inputs = self.tokenizer(queries, max_length=self.query_len, padding="max_length", truncation=True)
        ctx_inputs = self.tokenizer(ctxs, max_length=self.ctx_len, padding="max_length", truncation=True)

        return {
            "query_ids": torch.LongTensor(query_inputs["input_ids"]),
            "query_mask": torch.LongTensor(query_inputs["attention_mask"]),
            "ctx_ids": torch.LongTensor(ctx_inputs["input_ids"]),
            "ctx_mask": torch.LongTensor(ctx_inputs["attention_mask"]),
        }

