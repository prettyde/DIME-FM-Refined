from typing import Union, List

from transformers import AutoTokenizer
import torch


class HFPTTokenizer(object):
    def __init__(self, pt_name=None):

        self.pt_name = pt_name
        self.added_sep_token = 0
        self.added_cls_token = 0
        self.enable_add_tokens = False
        self.gpt_