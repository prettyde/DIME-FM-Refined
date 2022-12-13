from collections import OrderedDict
from socket import IP_DEFAULT_MULTICAST_LOOP
from typing import Tuple, Union, List
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import os


from .image_encoder.visual_transformer import visual_transformer_B32, visual_transformer_B16
# from .image_encoder.modified_resnet import modified_resnet_R50, modified_resnet_R101
from .text_encoder.text_transformer import text_transformers
from random import choice

from .clip import CLIP


BN = None

__all__ = ['filip_res50', 'filip_vitb32']

class FILIP(CLIP):
    def __init__(self,image_encode, text_encode, use_allgather, nn_size=2**16, nn_topk=1, \
                 return_dense=False, return_caption=False, return_nn_bank=False, text_mask_type=None, \
                 EDA=True, feature_dim=1024, embed_dim=768, forward_type='split', dense_mapping_image=2048, \
                 dense_mapping_language=512, dense_embed_dim=256, mask_rate=0.75, patch_number=14, \
                 text_mae_feature=False, return_simsiam=False, two_view=False, sparse=False, select_topk=False):
        super(FILIP, self).__init__(image_encode, text_encode, use_allgather)
        self.return_dense = return_dense
        self.return_caption = return_caption

        self.text_mask_type = text_mask_type
        self.select_topk = select_topk
        if self.return_dense:
            self.image_mapping = nn.Linear(dense_mapping_image, dense_embed_dim)
            self.text_mapping = nn.Linear(dense_mapping_language, dense_embed_dim)

        self.logit_scale_dense = nn.Parameter(torch.ones([]))
        nn.init.constant_(self.logit_scale_dense, np.log(1/0.07))

        if self.encode_text.text_encode_type == 'Transformer':
            self.sos_index = self.encode_text.tokenizer.encoder["<|startoftext|>"]
            self.padding_idx = 0
        if self.return_caption:
            self.caption_module = TransformerDecoderTextualHead(visual_feature_size=2048, vocab_size=self.encode_text.vocab_size, padding_idx=self.padding_idx)
        else:
            self.caption_module = None
        if text_mask_type is not None:
            enc_dim = self.encode_text.text_projection.weight.shape[-1]
            self.text_label_predictor = nn.Linear(enc_dim, self.encode_text.vocab_size)

    def encode_text_dense(self, texts, return_dense=True):
        text_features, word_features = self.encode_text(texts, return_dense=return_dense)
        word_features_d = self.text_mapping(word_features)
        return word_features_d

    def encode_image_dense(self, image):
        image_features, image_features_dense = self.visual(image.type(self.dtype), return_dense=True)
        image_features_dense = self.image_mapping(image_features_dense)
        return image_features_dense

    def encode_image(self, image, return_all=False):
        output = self.visual(image.type(self.dtype), return_dense=return_all)
   