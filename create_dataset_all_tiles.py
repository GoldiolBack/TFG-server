import os
import torch
import io
from torchvision.datasets.folder import default_loader
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import numpy as np


num_final_patches = 100
num_tiles = 44
num_final_patches_test = 30
num_test_tiles = 14
resize_im10 = 256
size_im20 = 256
resize_im20 = 128
channels10 = 4
channels20 = 6


f_test = open('S2_tiles_testing.txt', 'r')

HR_test = torch.zeros((num_test_tiles, num_final_patches_test, channels10, resize_im10, resize_im10), dtype=torch.float32)
LR_test = torch.zeros((num_test_tiles, num_final_patches_test, channels20, resize_im20, resize_im20), dtype=torch.float32)
target_test = torch.zeros((num_test_tiles, num_final_patches_test, channels20, size_im20, size_im20), dtype=torch.float32)

root_test_path = "/mnt/gpid07/users/oriol.esquena/patches_sentinel/test/"

k = 0

for i in f_test:
    im_name = i
    date = im_name[11:26]

    test_HR = "test_resized20_"+date+".npy"
    test_LR = "test_resized40_"+date+".npy"
    test_target = "real20_target_test_"+date+".npy"

    # read testing data from *.npy file
    HR_test_np = (np.load(root_test_path+test_HR))
    LR_test_np = (np.load(root_test_path+test_LR))
    target_test_np = (np.load(root_test_path+test_target))

    HR_test[k] = ((torch.from_numpy(HR_test_np)).permute(0, 3, 1, 2)).type(dtype=torch.float32)
    LR_test[k] = ((torch.from_numpy(LR_test_np)).permute(0, 3, 1, 2)).type(dtype=torch.float32)
    target_test[k] = ((torch.from_numpy(target_test_np)).permute(0, 3, 1, 2)).type(dtype=torch.float32)

    k += 1

HR_test = HR_test.reshape((num_test_tiles*num_final_patches_test, channels10, resize_im10, resize_im10))/2000
LR_test = LR_test.reshape((num_test_tiles*num_final_patches_test, channels20, resize_im20, resize_im20))/2000
target_test = target_test.reshape((num_test_tiles*num_final_patches_test, channels20, size_im20, size_im20))/2000

f_test.close()


f_train = open('S2_tiles_training.txt', 'r')

HR_data = torch.zeros((num_tiles, num_final_patches, channels10, resize_im10, resize_im10), dtype=torch.float32)
LR_data = torch.zeros((num_tiles, num_final_patches, channels20, resize_im20, resize_im20), dtype=torch.float32)
target_data = torch.zeros((num_tiles, num_final_patches, channels20, size_im20, size_im20), dtype=torch.float32)

root_train_path = "/mnt/gpid07/users/oriol.esquena/patches_sentinel/train/"

k = 0

for i in f_train:
    im_name = i
    date = im_name[11:26]

    file_name_HR = "input10_resized20_"+date+".npy"
    file_name_LR = "input20_resized40_"+date+".npy"
    file_name_target = "real20_target_"+date+".npy"

    # read training data from *.npy file
    HR_data_np = (np.load(root_train_path+file_name_HR))
    LR_data_np = (np.load(root_train_path+file_name_LR))
    target_data_np = (np.load(root_train_path+file_name_target))

    HR_data[k] = ((torch.from_numpy(HR_data_np)).permute(0, 3, 1, 2)).type(dtype=torch.float32)
    LR_data[k] = ((torch.from_numpy(LR_data_np)).permute(0, 3, 1, 2)).type(dtype=torch.float32)
    target_data[k] = ((torch.from_numpy(target_data_np)).permute(0, 3, 1, 2)).type(dtype=torch.float32)

    k += 1

HR_data = HR_data.reshape((num_tiles*num_final_patches, channels10, resize_im10, resize_im10))/2000
LR_data = LR_data.reshape((num_tiles*num_final_patches, channels20, resize_im20, resize_im20))/2000
target_data = target_data.reshape((num_tiles*num_final_patches, channels20, size_im20, size_im20))/2000

f_train.close()


class PatchesDataset(Dataset):
    """Patches dataset."""

    def __init__(self, hr, lr, target, transform=None):
        """
        Args:
            HRdata: images of high resolution
            LRdata: images of low resolution
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.hr = hr
        self.lr = lr
        self.target = target
        self.transform = transform

    def __len__(self):
        return len(self.hr)

    def __getitem__(self, idx):

        hr_data = self.hr[idx]
        lr_data = self.lr[idx]
        t_data = self.target[idx]

        if self.transform is not None:
            hr_data = self.transform(self.hr)
            lr_data = self.transform(self.lr)
            t_data = self.transform(self.target)

        return hr_data, lr_data, t_data


set_ds = PatchesDataset(HR_data, LR_data, target_data)
train_ds, val_ds = torch.utils.data.random_split(set_ds, [90*num_tiles, 10*num_tiles])
test_ds = PatchesDataset(HR_test, LR_test, target_test)
