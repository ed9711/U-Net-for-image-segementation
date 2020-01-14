from PIL import Image
from PIL import ImageSequence
import glob
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import cv2


class GetData(Dataset):

    def __init__(self, image_path, mask_path):
        self.masks_path = glob.glob(str(mask_path) + "/*")
        self.images_path = glob.glob(str(image_path) + str("/*"))
        self.data_len = len(self.images_path)

    def __getitem__(self, index):
        single_mask = self.masks_path[index]
        single_img = self.images_path[index]
        img = Image.open(single_img)  # .convert('L')
        img = img.resize((128, 128))
        img_np = np.array(img)
        img_np = img_np.transpose((2, 0, 1))

        img_list = [image / 255 for image in img_np]

        mask = Image.open(single_mask)
        mask = mask.resize((128, 128))
        mask_np = np.array(mask)
        mask_list = [image / 255 for image in mask_np]
        # print(img_np.shape)
        # print(mask_np.shape)

        img_tensors = torch.Tensor(img_list).float()
        mask_tensors = torch.Tensor(mask_list).long()

        return (img_tensors, mask_tensors, mask_np)

    def __len__(self):
        return self.data_len
