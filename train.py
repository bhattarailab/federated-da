import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

import numpy as np
import wandb

from unet import Encoder, Decoder
from dataset import SegDataset
from eval import eval_net
from utils import *
import copy
from pathlib import Path
#change here
device=0


set_random_seed(0)
model_dir = 'saved_models/'
batch_size_train = 16
batch_size_val = 1
train_num = 0
val_num = 0
eval_frq = 1000

save_frq = 20000 # save the model every 50000 iterations
total_iters = 100001

##specify folder that contains the dataset and make changes on utils.py get_dir_lists() function
dr = '/raid/binod/projects/multicenter/data'

data_root = Path(dr)

##make changes on utils.py get_dir_list() to change data paths
train_dir_list, val_dir_list = get_dir_lists(data_root)

# print(f'Length--train_data: {len(train_dir_list)}---Length--train_data: {len(val_dir_list)}')

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

#for logging

# wandb.login()

crossentropy_loss = nn.CrossEntropyLoss()

model_name = 'UNet'

mkdir(model_dir)

#dataset

train_dataset = SegDataset(train_dir_list, True)
val_dataset = SegDataset(val_dir_list, False)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, num_workers=4, pin_memory=True)
eval_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False, num_workers=1, pin_memory=False, drop_last=True)


encoder = Encoder(3, 2, bilinear=False)
decoder = Decoder(3, 2, bilinear=False)


if torch.cuda.is_available():
    
# load from checkpoints
    # encoder.load_state_dict(torch.load('/raid/binod/projects/multicenter/expts/be_scratch/saved_models/UNet_best_encoder_2000.pth', map_location='cpu'))
    encoder.to(device)
    encoder.train()

    ## uncomment to Load Decoder checkpoint trained on data ####

    # decoder.load_state_dict(torch.load('/raid/binod/projects/multicenter/expts/be_scratch/saved_models/UNet_best_decoder_2000.pth', map_location='cpu'))
    decoder.to(device)
    
    #comment if pretrained decoder is trained
    decoder.train()

    
    #uncomment if pretrained decoder is used for evaluation
    # decoder.eval()

    



#wandb

# wandb.init(project='try', entity='try', name='try')
# wandb.watch([net], log='all')

# ------- 4. define optimizer --------
print("---define optimizer...")

optimizer_enc  = optim.Adam(encoder.parameters(), lr=0.0002, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-7)
scheduler_enc = optim.lr_scheduler.MultiStepLR(optimizer_enc, [25000, 50000, 75000], 0.5)


#comment both lines if pretrained decoder is used for evaluation 
optimizer_dec  = optim.Adam(decoder.parameters(), lr=0.0002, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-7)
scheduler_dec = optim.lr_scheduler.MultiStepLR(optimizer_dec, [25000, 50000, 75000], 0.5)

# ------- 5. training process --------
print("---start training...")
ite_num = 0
running_loss = 0.0

print(f'---------Number of training images: {len(train_dir_list)}--------')
print(f'---------Number of val images: {len(val_dir_list)}--------')


results = open('results.txt', 'w')
print("Writing to a file")
results.write('Iter\t MEAN\n')

max_miou = 0
best_ite = 0

best_encoder = None

#comment to use pretrained decoder
best_decoder = None


while True:
    
    for i, data in enumerate(train_dataloader):
        ite_num = ite_num + 1

        inputs, labels = data['image'].to(device), data['mask'].to(device)

        optimizer_enc.zero_grad()

        #comment if pretrained decoder is used for evaluation
        optimizer_dec.zero_grad()

        x1, x2, x3, x4, x5 = encoder(inputs)

        # for p in decoder.parameters():
        #     p.requires_grad = False

        mask_pred = decoder(x1, x2, x3, x4, x5)

        loss = crossentropy_loss(mask_pred, labels.type(dtype=torch.long))

        loss.backward()
        optimizer_enc.step()
        scheduler_enc.step()

    #comment both lines below if pretrained decoder is used for evaluation only
        optimizer_dec.step()
        scheduler_dec.step()

#         print(f'Iter: {ite_num}\t Loss CE: {loss.item()}')

#uncomment line below to enable logging in wandb
 #       wandb.log({"Train/CE_loss":loss.item()}, step=ite_num)
  
        if ite_num % save_frq == 0:
            print('saving_model')
            torch.save(encoder.state_dict(), model_dir + model_name+"_encoder_%d.pth" % (ite_num))
            
            
            #comment if pretrained decoder is used for evaluation only
            torch.save(decoder.state_dict(), model_dir + model_name+"_decoder_%d.pth" % (ite_num))


        if ite_num % eval_frq == 0:
            print('Validating')
         
            ce_loss_val, fold_iou = eval_net(encoder, decoder, eval_dataloader, device, ite_num)

            if fold_iou.item() > max_miou:
                max_miou = fold_iou.item()
                
                best_ite = ite_num
                best_encoder = copy.deepcopy(encoder)
                
                #comment if pretrained decoder is used for evaluation only
                best_decoder = copy.deepcopy(decoder)

            print(f'Ite: {ite_num}\t Val_loss_ce: {ce_loss_val}\t MeanIOU: {fold_iou.item()}')


        if total_iters <= ite_num:
            torch.save(best_encoder.state_dict(), model_dir + model_name+"_best_encoder_%d.pth" % (best_ite))
            
            #comment if pretrained decoder is used for evaluation only
            torch.save(best_decoder.state_dict(), model_dir + model_name+"_best_decoder_%d.pth" % (best_ite))
            
            np.array([best_ite, max_miou]).tofile(results, sep="\t")
            results.write('\n')
            results.close()        
            break
  
    if total_iters <= ite_num:
        break
