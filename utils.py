import random
import glob
import torch
import numpy as np

def get_dir_lists(data_root):

    #be
    # train_dir_be = data_root / 'BE' / 'train' / 'image'
    # val_dir_be =  data_root / 'BE' / 'val' / 'image'

    # train_dir_list = list(train_dir_be.glob('*.png'))
    # val_dir_list =  list(val_dir_be.glob('*.png'))

   #polyp
    train_dir_polyp = data_root / 'Polyp' / 'train' /'image'
    val_dir_polyp =  data_root / 'Polyp' /'val' / 'image'

    train_dir_list = list(train_dir_polyp.glob('*.jpg'))
    val_dir_list =  list(val_dir_polyp.glob('*.jpg'))

    #both
    # train_dir_be = data_root / 'BE' / 'train' / 'image'
    # val_dir_be =  data_root / 'BE' / 'val' / 'image'
    # train_dir_polyp = data_root / 'Polyp' / 'train' /'image'
    # val_dir_polyp =  data_root / 'Polyp' / 'val' / 'image'
    # train_dir_list = list(train_dir_be.glob('*.png')) + list(train_dir_polyp.glob('*.jpg'))
    # val_dir_list = list(val_dir_be.glob('*.png')) + list(val_dir_polyp.glob('*.jpg'))
   
   #polyp
    
    return train_dir_list, val_dir_list

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

