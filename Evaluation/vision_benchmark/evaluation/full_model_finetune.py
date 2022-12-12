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

        input_dim, output_dim = config.MODEL.SPEC.EMBED_DIM, config.DATASET.NUM_CLASSES
        self.optim = None
        self.l2_lambda = l2_lambda
        self.channel_bn = torch.nn.BatchNorm1d(
            input_dim,
            affine=False,
        )
        self.layers = torch.nn.Sequential(torch.nn.Linear(input_dim, output_dim))

        if config.TRAIN.INIT_HEAD_WITH_TEXT_ENCODER:
            if config.MODEL.SPEC.TEXT.TOKENIZER == 'clip':
                tokenizer = SimpleTokenizer()
            elif 'hf_' in config.MODEL.SPEC.TEXT.TOKENIZER:
                tokenizer = HFPTTokenizer(pt_name=config.MODEL.SPEC.TEXT.TOKENIZER[3:])
            else:
                tokenizer = None

            zeroshot_weights = extract_text_features(config, tokenizer, model=self.backbone, return_numpy=False)
            self.layers[0].weight.data = zeroshot_weights.T.to(self.layers[0].weight.dtype).to(self.layers[0].weight.device).contiguous()
            self.layers[0].bias.data.fill_(0.0)

        if config.TRAIN.MERGE_ENCODER_AND_HEAD_PROJ and self.backbone.visual.proj is not None:
            encoder_proj = self.backbone.visual.proj
            head_proj = self.layers[0].weight.data
            head_bias = self.layers[0].bias.data
            self.backbone.visual.proj = None
            encoder_ic, encoder_oc = encoder_proj.shape
            self.channel_bn = torch.nn.BatchNorm1d(
                encoder_ic,
                affine=False,
            )
            self.layers = torch.nn.Sequential(torch.nn.Linear(encoder_ic, output_dim))
            self.layers[0].weight.data = head_proj @ encoder_proj.T.to(head_proj.dtype).to(head_proj.device)
            self.layers[0].bias.data = head_bias

        self.logit_scale = nn.Parameter(torch.ones([]))
        self.logit_scale.requires_grad = config.TRAIN.TRAINABLE_LOGIT_SCALE
        if config.TRAIN.LOGIT_SCALE_INIT == 'pretrained':
            self.logit_scale.data = self.backbone.logit_scale.data.to(self.logit_scale.dtype).to(self.logit_scale.device)
        elif config.TRAIN.LOGIT_SCALE_INIT == 'ln_cls':
            self.logit_scale.data *= np.log(np.log(config.DATASET.NUM_CLASSES))
        elif config.TRAIN.LOGIT_SCALE_INIT == 'clip':
            self.logit_scale.data *= np.log(1 / 0.07)
        else:
            self.logit_scale.data *= 0

        self.normalize_visual_output = config.TRAIN.NORMALIZE_VISUAL_FEATURE

        if not config.TRAIN.USE_CHANNEL_BN:
            self.channel_bn = nn.Identity()

    def forward(self, img):
        pdtype = img.dtype
        feature = self.backbone(img).to(pdtype)
        outputs = self.channel_bn(feature)

        if self.normalize_visual_output:
            outputs = F.normalize(outputs)

        outputs = self.logit_scale.exp() * self.layers(outputs)
        return outputs


def hyperparameter_sweep(train_dataloader, val_dataloader, config):
    logging.info(f"=> Learning rate {config.TRAIN.LR}: tuning l2 regularization strength.")
    start = time.time()
    l2_lambda_list = np.logspace(config.TRAIN.SEARCH_WD_LOG_LOWER, config.TRAIN.SEARCH_WD_LOG_UPPER, num=97).tolist()
    l2_lambda_init_idx = [i for i, val in enumerate(l2_lambda_list) if val in set(np.logspace(config.TRAIN.SEARCH_WD_LOG_LOWER, config.TRAIN.SEARCH_WD_LOG_UPPER, num=7))]
    peak_idx = -1
    peak_score = 0
    iter_num = 0
    for idx in l2_lambda_init_idx:
        config.defrost()
        config.TRAIN.WD = l2_lambda_list[idx]

        try:
            best_score_ = train_task(train_dataloader, val_dataloader, config, sweep_run=True)
        except:
            best_score_ = 0.0
            gpu_gc()
            continue       

        if best_score_ > peak_score:
            peak_idx = idx
            peak_score = best_score_
    logging.info(f"Iteration {iter_num}: l2_lambda: {l2_lambda_list[peak_idx]}, best score {best_score_}")

    step_span = 8
    while step_span > 0:
        left, right = max(peak_idx - step_span, 0), min(peak_idx + step_span, len(l2_lambda_list) - 1)
        search_idx = []
        if left != peak_idx:
            search_idx.append(left)
        if right != peak_idx:
            search_idx.append(right)
        for idx in search_idx:
            # WD_SEARCH_LEFT is used in the inital release, whereas we later find WD_SEARCH_IDX to be more stable.
            if config.TRAIN.WD_SEARCH_LEFT:
                config.TRAIN.WD = l2_lambda_list[left]
            else:
                config.TRAIN.WD = l2_lambda_list[idx]

            try:
                best_score_ = train_task(train_dataloader, val_dataloader, config, sweep_run=True)
            except:
                best_score_ = 0.0
                gpu_gc()
                continue

            if best_score_ > peak_score:
                peak_idx = idx
                peak_score = best_score_
        iter_num += 1
        logging.info(f"Iteration {iter_num}: l2_lambda: {l2_lambda_list[peak_idx]}, best score {best_score_}")
        step_span //= 2

    logging.info(f"=> Learning rate {config.TRAIN.LR}: The best l2 lambda is {l2_lambda_list[peak_idx]}")
    logging.info('=> Learning rate {}: l2 regularization strength tuning duration time: {:.2f}s'.format(config.TRAIN.LR, time.time() - start))
    return l2_lambda_list[peak_idx], peak_score


def clone_loader(loader, shuffle=True):

    return create_dataloader(
        loader.dataset,
        batch_size=loader.batch_size,
        shuffle=shuffle,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
    )


def train_task(train_dataloader, test_dataloader, config, sweep_run=False):
    best_acc1 = 0

    model = Classifier(config, 0)
    pytorch_total_pa