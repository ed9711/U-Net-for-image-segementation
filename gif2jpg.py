from PIL import Image
from PIL import ImageSequence
import glob
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import cv2

masks_path = glob.glob(str('./car_data/train/mask/1') + "/*")
# images_path = glob.glob(str('./car_data/train/input') + str("/*"))

for index in range(len(masks_path)):
    single_mask = masks_path[index]
    # single_img = images_path[index]
    print(single_mask)
    # img = Image.open(single_img)
    # img = img.resize((128, 128))
    mask = Image.open(single_mask)
    mask = mask.resize((128, 128))
    # img.save("{}.jpg".format(single_img.strip(".gif")))
    result = single_mask.split('\\')

    mask.convert('RGB').convert('L').save("./car_data/train/mask/{}.jpg".format(result[1].strip(".gif")))
