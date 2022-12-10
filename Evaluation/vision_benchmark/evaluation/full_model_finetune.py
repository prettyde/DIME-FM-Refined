"""
Linear classifier implemented with Pytorch Linear class
"""

import os
from re import L
import time
import logging
import pickle
import numpy as np
import sys, json
import random

from torch import nn
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from .feature import FeatureData, get_model
from ..optim import build_optimizer
from ..evaluation.metric import get_metric

from ..common.constants import get_dataset_hub, VISION_DATASET_STORAGE
from ..models import *
from ..datasets import class_map, template_map

from vision_benchmark.datasets import SimpleTokenizer, HFPTTokenizer
from vision_benchmark.evaluation import clip_zeroshot_evaluator, construct_dataloader

import pdb

from tqdm import tqdm
from vision_datasets import ManifestDataset
from nltk.corpus import wordnet as wn
import nltk
from nltk.tokenize import word_tokenize
from .feature import extract_text_features, create_dataloader

import gc

nltk.download('punkt')
nltk.download('wordnet')


MULTILABEL_DATASETS = {"voc-2007-classification","chestx-ray8"}


def gpu_gc():
    gc.collect()
    torch.cuda.empty_cache()

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Classifier(torch.nn.Module):
    """
    Linear classifier.
    """

    def __init__(self, config, l2_lambda):
        super(Classifier, self).__init__()

        self.backbone = get_model(config, feature_type="image")

        if config.MODEL.NAME.startswith('vit_'):
            self.backbone.head = self.backbone.head_dist = None

        for name, param in self.backbone.named_parameters():
            if name.startswith('text') or name.startswith('transformer') or name.startswith('token_embedding') or name.startswith('ln_final') or name.startswith('positional_embedding') or name.startswith('logit_scale'):
                param.requires_grad = False

            if config.TRAIN.FREEZE_IMAGE_BACKBONE:
                # freeze for {supervised ViT, MAE, MoCov3} under linear probing settings
                for model_keyword in ['vit', 'mae', 'mocov3']:
                    if config.MODEL.NAME.startswith(f'{model_keyword}_'):
                        param.requires_grad = False

                if name.startswith('visual.conv1') or name.startswith('visual.ln_pre') or name.startswith('visual.transformer') or name.startswith('visual'):
                    param.requires_grad = False

        input_dim, output_dim = config.MO