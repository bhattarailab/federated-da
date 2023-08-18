import torch
import torch.nn as nn
import os
import cv2
import numpy as np
import torch.nn.functional as F
from PIL import Image

from metrics import IoU
from tqdm import tqdm

from utils import *


crossentropy_loss = nn.CrossEntropyLoss()


#metric for iou
def compute_mean_iou(pred, label):

    unique_labels = np.unique(label)
    num_unique_labels = len(unique_labels)

    I = np.zeros(num_unique_labels)
    U = np.zeros(num_unique_labels)

    for index, val in enumerate(unique_labels):
        pred_i = pred == val
        label_i = label == val

        I[index] = float(np.sum(np.logical_and(label_i, pred_i)))
        U[index] = float(np.sum(np.logical_or(label_i, pred_i)))


    mean_iou = np.mean(I / U)
    return mean_iou


#-----------------------------------------#
def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def eval_net(encoder, decoder, loader, device, current_step):
    """Evaluation without the densecrf with the dice coefficient"""
    encoder.eval()
    decoder.eval()
    n_val = len(loader)  # the number of batch
    loss_ce = 0

    miou_list = []

    for batch in tqdm(loader):
        imgs, true_masks, idx = batch['image'], batch['mask'], batch['idx'][0]

        imgs = imgs.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device)


        img_path = os.path.join('val_results', idx)
        mkdirs(img_path)

        with torch.no_grad():
            x1, x2, x3, x4, x5 = encoder(imgs)
            mask_pred = decoder(x1, x2, x3, x4, x5)
            # mask_pred= net(imgs)

        loss = crossentropy_loss(mask_pred, true_masks.type(dtype=torch.long))
        loss_ce += loss

        pred = torch.argmax(mask_pred, 1)

        miou = compute_mean_iou(pred.squeeze().cpu().numpy().flatten().astype(np.uint8), true_masks.squeeze().cpu().numpy().flatten().astype(np.uint8))
        miou_list.append(miou)
        
        if current_step % 25000 == 0:
            pred_img = pred.squeeze().float().cpu()*255

            temp_inp = imgs.squeeze().float().cpu().numpy().transpose(1,2,0)*255

            gt_mask = true_masks.squeeze().float().cpu()*255
            border = np.ones((gt_mask.shape[0], 5))*255
            imgs_comb = np.hstack((pred_img, border.astype(np.uint8),gt_mask))
            cv2.imwrite(os.path.join(img_path, f'{idx}__{current_step}.png'), cv2.cvtColor(imgs_comb, cv2.COLOR_RGB2BGR))

    encoder.train()
    
    return loss_ce/n_val, np.mean(miou_list)
