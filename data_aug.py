from PIL import Image
import numpy as np
from random import randint
import matplotlib.pyplot as plt


def data_aug(image_path, mask_path, images_path, masks_path, choice):
    counter = 0
    for i in range(len(images_path)):
        single_mask = masks_path[i]
        single_img = images_path[i]
        img = Image.open(single_img)
        mask = Image.open(single_mask)

        if choice == 0:
            # flip
            img = img.resize((128, 128))
            img_np = np.array(img)
            flip_num = randint(0, 1)
            img_np = np.flip(img_np, flip_num)

            mask = mask.resize((128, 128))
            mask_np = np.array(mask)
            mask_np = np.flip(mask_np, flip_num)

            aug = "flip"

        elif choice == 1:
            # Gaussian_noise
            img = img.resize((128, 128))
            img_np = np.array(img)
            noise = np.random.normal(0, 1, img_np.shape)
            img_np = img_np.astype("int16")
            img_np = img_np + noise
            img_np[img_np > 255] = 255
            img_np[img_np < 0] = 0
            img_np = img_np.astype("uint8")

            mask = mask.resize((128, 128))
            mask_np = np.array(mask)

            aug = "noise"

        elif choice == 2:
            # Brightness
            img = img.resize((128, 128))
            img_np = np.array(img)
            pix_add = randint(-20, 20)
            img_np = img_np.astype("int16")
            img_np = img_np + pix_add
            img_np[img_np > 255] = 255
            img_np[img_np < 0] = 0
            img_np = img_np.astype("uint8")

            mask = mask.resize((128, 128))
            mask_np = np.array(mask)
            aug = "bright"

        elif choice == 3:
            # Crop
            img = img.resize((128, 128))
            mask = mask.resize((128, 128))
            area = (20, 20, 100, 100)
            img = img.crop(area)
            mask = mask.crop(area)
            img = img.resize((128, 128))
            mask = mask.resize((128, 128))
            # img_np = np.array(img)
            # mask_np = np.array(mask)
            img.save("{}/{}crop.jpg".format(image_path, counter + i))
            mask.save("{}/{}crop.jpg".format(mask_path, counter + i))
            continue

        # plt.imshow(img_np)
        # plt.show()
        # plt.imshow(mask_np, cmap=plt.cm.gray)
        # plt.show()

        result_img = Image.fromarray(img_np, 'RGB')
        result_mask = Image.fromarray(mask_np, 'L')
        result_img.save("{}/{}{}.jpg".format(image_path, counter + i, aug))
        result_mask.save("{}/{}{}.jpg".format(mask_path, counter + i, aug))
