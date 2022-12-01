from typing import Union, List

from transformers import AutoTokenizer
import torch


class HFPTTokenizer(object):
    def __init__(self, pt_name = None):
        
        self.pt_name = pt_name
        self.added_sep_token = 0
        self.added_cls_token = 0
        self.enable_add_tokens = False
        self.gpt_special_case = ((not self.enable_add_tokens) and ('gpt' in self.pt_name))

        if (pt_name is None):
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(pt_name)

        # Adding tokens to GPT causing NaN training loss.  
        # Disable for now until further investigation. 
        if (self.enable_add_tokens):
            if (self.tokenizer.sep_token is None):
                self.tokenizer.add_special_tokens({'sep_token': '<SEP>'})
                self.added_sep_token = 1
    
            if (self.tokenizer.cls_token is None):
                self.tokenizer.add_special_tokens({'cls_token': '<CLS>'})
                self.added_cls_token = 1

        if (self.gpt_special_case):
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.sep_token = self.tokenizer.eos_token

    def get_eot_token(self):
        return self.tokenizer.encode(self.tokenizer.sep_token, add_special_tokens=False)[0]

    def get_sot_token(self):
        return self.tokenizer.encode(self.tokenizer.cls_token, add_special_tokens=False)[0]

    def get_eot_token_list(self):
        return self.tokenizer.encode(self.tokenizer.sep_token, add_special_tokens=False)

    def get_sot_token_list(self):
        return self.tokenizer.encode(self.tokenizer.cls_token, add_special_tokens=False)

    def get_tokenizer_obj(self):
        return self.tokenizer

    # Language model needs to know if new tokens
    # were added to the dictionary.
    def check_added_tokens(self):
        return self.added_sep_token + self.added_cls_token

    def tokenize(self, texts: Union[str, List[str]], context_length: int = 77):
        if isinstance(texts, str):
            texts = [texts]

 