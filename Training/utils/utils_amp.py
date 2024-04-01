import sys
sys.path.insert(0, '../model_distill/')
import torch
import torch.nn as nn
import time
from .utils import step_update, AverageMeter, zero_grad, get_loss, reduce_mean


def scaler_step(objs, scaler):
    for k, v in objs.items():
        if v is not None:
            scaler.step(v)
    return objs


def train(models, dataloaders, optimizers, schedulers, scaler, args, epoch, writer, text_train_enumerator=None, paired_train_enumerator=None):
    losses = AverageMeter()
    vl_losses = AverageMeter()
    pvl_losses = AverageMeter()
    udist_losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    if text_train_enumerator is None:
        text_train_enumerator = enumerate(dataloaders['text_train_dataloader'])
    models['student_img_model'].train()

    if args.local_rank == -1:
        kl_loss = nn.KLDivLoss(r