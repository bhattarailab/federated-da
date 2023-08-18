from os.path import splitext
import os
from os import listdir
import numpy as np
import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
from torchvision import transforms
# from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms.functional as tf


def get_hog_f(img):
#     print(img.shape)
#     print(type(img))
#     img_copy = image.copy()
#         hog_feature = hog(img_copy.resize((64, 128)), orientations=4, pixels_per_cell=(), cells_per_block=(2, 2), visualize=False, multichannel=True)
#     ip = img.numpy().transpose(1,2,0)
#     print(f'Transposed Img {ip.shape}')
    hog_f = hog(img.resize((64, 128)), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False, multichannel=True)
    return torch.from_numpy(hog_f)


class SegDataset(Dataset):
    def __init__(self, files_name, is_train):
#         self.imgs_dir = imgs_dir
#         self.masks_dir = masks_dir
        self.files_name = files_name
        self.is_train = is_train


    def __len__(self):
        return len(self.files_name)

    @classmethod
    def preprocess(cls, image, mask, is_train):
        
        rn = torch.rand(1).item()  

        if is_train:
            image = transforms.Resize((418, 418), Image.NEAREST)(image)
            mask = transforms.Resize((418, 418), Image.NEAREST)(mask)      
        
        if rn > 0.15 and is_train:  
            #No resize 
            
#             print('Random transform')
            rn = torch.rand(1).item()
            if rn>0.2:
                #random rotation
                rotation_params = transforms.RandomRotation.get_params([-45,45])
                image = transforms.functional.rotate(image, rotation_params)
                mask = transforms.functional.rotate(mask, rotation_params)
#             print(rotation_params)
            #random flip
            rn = torch.rand(1).item()
            if rn>0.3: 
                image = transforms.functional.hflip(image)
                mask = transforms.functional.hflip(mask)
            
            rn = torch.rand(1).item()
            if rn>0.4: 
                image = transforms.functional.vflip(image)
                mask = transforms.functional.vflip(mask)
                
            #adjust brightness 
            # scale = torch.rand(1).item()/5
            # if torch.rand(1).item()>0.5:
            #     image = tf.adjust_brightness(image, 1-scale)
                   
            # else:
            #     image = tf.adjust_brightness(image, 1+scale)
                
            # #adjust contrast
            # scale = torch.rand(1).item()/10
            # if torch.rand(1).item()>0.5:
            #     image = tf.adjust_contrast(image, 1-scale)
                   
            # else:
            #     image = tf.adjust_contrast(image, 1+scale)
                
#             #adjust saturation
#             rn = torch.rand(1).item()
#             if rn<0.2:
#                 scale = rn + 1
#                 image = tf.adjust_saturation(image, scale)
#             elif rn>0.8:
#                 image = image = tf.adjust_saturation(image, rn)
                    
                    
        #Crop
        if is_train:
            rn = torch.rand(1).item()         
            if rn<0.8:
                #Random Crop
                crop_params = transforms.RandomCrop.get_params(image, (256, 256))

        #         print(crop_params)      
                image = transforms.functional.crop(image, *crop_params)
                mask = transforms.functional.crop(mask, *crop_params)

            else:
                #Center Crop
                image = transforms.CenterCrop(256)(image)
                mask = transforms.CenterCrop(256)(mask)
#         else:
#             image = transforms.Resize(256, InterpolationMode.NEAREST)(image)
#             mask = transforms.Resize(256, InterpolationMode.NEAREST)(mask)
                    
        to_tensor = transforms.ToTensor()        
        image = to_tensor(image)     
        mask = np.array(mask) / 255

        # print(np.unique(mask))

#       mask = np.expand_dims(mask, 0)
        mask = torch.from_numpy(mask)
        
        return image, mask
        


    def __getitem__(self, idx):

        image_file_name = self.files_name[idx]
        img_file = str(image_file_name)
        mask_file = str(image_file_name).replace('image', 'label')  
        img = Image.open(img_file).convert("RGB")
        mask = Image.open(mask_file).convert('L')

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img, mask  = self.preprocess(img, mask, self.is_train)
#         mask = self.preprocess(mask, self.scale)
        
        return {
            'image': img.type(torch.FloatTensor),
            'mask': mask,
            # 'mask': mask.type(torch.FloatTensor),
            'idx' : img_file.split('/')[-1].split('.')[0]
        }
