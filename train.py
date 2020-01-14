import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable
from dice_coefficient import *
from draw_outline import *


def accuracy(mask, prediction):
    ims = [mask, prediction]
    np_ims = []
    for item in ims:
        if 'str' in str(type(item)):
            item = np.array(Image.open(item))
        elif 'PIL' in str(type(item)):
            item = np.array(item)
        elif 'torch' in str(type(item)):
            item = item.numpy()
        np_ims.append(item)

    compare = np.equal(np_ims[0], np_ims[1])
    accuracy = np.sum(compare)

    return accuracy / len(np_ims[0].flatten()) # dice score, true positive/(true positve+false positive+false negetive)


def train_net(trainLoad, cnn, optimizer, epoch, name, device):
    save_dir = "./outputs1/"
    epoch_list = []
    for i in range(epoch):
        cnn = torch.nn.DataParallel(cnn, device_ids=list(
            range(torch.cuda.device_count()))).cuda()
        cnn.train()
        for batch, (images, masks, original_masks) in enumerate(trainLoad):
            images = Variable(images.cuda())
            masks = Variable(masks.cuda())
            # images.to(device=device)
            # masks.to(device=device)
            outputs = cnn(images)
            # print(outputs.shape, masks.shape)
            if name == 0:
                loss = nn.CrossEntropyLoss()
                loss = loss(outputs, masks)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            elif name == 1:
                loss1 = DiceCoefficient()
                # print(outputs.squeeze(dim=0)[0].shape)
                loss = loss1.apply(outputs.squeeze(dim=0)[0], masks.squeeze(dim=0).float())
                loss += loss1.apply(outputs.squeeze(dim=0)[1], masks.squeeze(dim=0).float())
                loss += loss1.apply(outputs.squeeze(dim=0)[2], masks.squeeze(dim=0).float())
                loss = loss/3
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        cnn.eval()
        total_acc = 0
        total_loss = 0
        for batch, (images, masks, original_masks) in enumerate(trainLoad):
            with torch.no_grad():
                images = Variable(images.cuda())
                masks = Variable(masks.cuda())
                # images.to(device=device)
                # masks.to(device=device)
                outputs = cnn(images)
                if name == 0:
                    loss = nn.CrossEntropyLoss()
                    loss = loss(outputs, masks)
                elif name == 1:
                    loss1 = DiceCoefficient()
                    # print(outputs.squeeze(dim=0)[0].shape)
                    loss = loss1.apply(outputs.squeeze(dim=0)[0], masks.squeeze(dim=0).float())
                    loss += loss1.apply(outputs.squeeze(dim=0)[1], masks.squeeze(dim=0).float())
                    loss += loss1.apply(outputs.squeeze(dim=0)[2], masks.squeeze(dim=0).float())
                    loss = loss / 3
                preds = torch.argmax(outputs, dim=1).float()
                acc = 0
                for j in range(images.size()[0]):
                    acc += accuracy(masks.cpu()[j], preds.cpu()[j])
                total_acc += acc / images.size()[0]
                total_loss = total_loss + loss.cuda().item()
        epoch_list.append([total_acc / (batch + 1), total_loss / (batch + 1)])
        print("Epoch: {}, Train loss: {}, Train acc: {}".format(i, total_loss / (batch + 1), total_acc / (batch + 1)))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if name == 0:
        name = "CEL"
    else:
        name = "DICE"
    torch.save(cnn, save_dir + "/model_lossF_{}.pwf".format(name))
    # name == 0 is CrossEntropyLoss, name == 1 is Dice loss
    return cnn, save_dir + "/model_lossf_{}.pwf".format(name)


def division_array(crop_size, crop_num1, crop_num2, dim1, dim2):
    div_array = np.zeros([dim1, dim2])  # make division array
    one_array = np.ones([crop_size, crop_size])  # one array to be added to div_array
    dim1_stride = int((dim1 - crop_size) / (crop_num1))  # vertical stride
    dim2_stride = int((dim2 - crop_size) / (crop_num2))  # horizontal stride
    for i in range(crop_num1):
        for j in range(crop_num2):
            div_array[dim1_stride * i:dim1_stride * i + crop_size,
            dim2_stride * j:dim2_stride * j + crop_size] += one_array
    return div_array


def image_concatenate(image, crop_num1, crop_num2, dim1, dim2):
    crop_size = image.shape[1]  # size of crop
    empty_array = np.zeros([dim1, dim2]).astype("float64")  # to make sure no overflow
    dim1_stride = int((dim1 - crop_size) / (crop_num1))  # vertical stride
    dim2_stride = int((dim2 - crop_size) / (crop_num2))  # horizontal stride
    index = 0
    for i in range(crop_num1):
        for j in range(crop_num2):
            # add image to empty_array at specific position
            empty_array[dim1_stride * i:dim1_stride * i + crop_size,
            dim2_stride * j:dim2_stride * j + crop_size] += image[index]
        index += 1
    return empty_array


def save_prediction_image(stacked_img, im_name, epoch, save_folder_name="result_images", save_im=True):
    div_arr = division_array(128, 1, 1, 128, 128)
    img_cont = image_concatenate(stacked_img.cpu().data.numpy(), 1, 1, 128, 128)
    img_cont = img_cont / div_arr
    img_cont[img_cont >= 0.5] = 1
    img_cont[img_cont < 0.5] = 0
    img_cont = img_cont * 255
    img_cont_np = img_cont.astype('uint8')
    img_cont = Image.fromarray(img_cont_np)
    # organize images in every epoch
    desired_path = save_folder_name + '/epoch_' + str(epoch) + '/'
    # Create the path if it does not exist
    if not os.path.exists(desired_path):
        os.makedirs(desired_path)
    # Save Image!
    export_name = str(im_name) + '.png'
    img_cont.save(desired_path + export_name)
    return img_cont_np


def test_model(model, data_val, epoch, i, make_prediction=True, save_folder_name='result'):
    model = torch.load(model)
    model = torch.nn.DataParallel(model, device_ids=list(
        range(torch.cuda.device_count()))).cuda()
    model.eval()
    total_val_loss = 0
    total_val_acc = 0
    for batch, (images_v, masks_v, original_msk) in enumerate(data_val):
        stacked_img = torch.Tensor([]).cuda()
        for index in range(images_v.size()[1]):
            with torch.no_grad():
                image_v = Variable(images_v.cuda())
                mask_v = Variable(masks_v.cuda())

                output_v = model(image_v)
                # print(output_v.squeeze(dim=0).shape)
                total_val_loss = total_val_loss + dice_coeff(output_v.squeeze(dim=0), mask_v).cpu().item()
                output_v = torch.argmax(output_v, dim=1).float()
                stacked_img = torch.cat((stacked_img, output_v))
        if make_prediction:
            pred_msk = save_prediction_image(stacked_img, batch, epoch * (i + 1), save_folder_name)
            drawOutline(images_v, pred_msk, batch)
            acc_val = accuracy(original_msk, pred_msk)
            total_val_acc = total_val_acc + acc_val

    return total_val_acc / (batch + 1), total_val_loss / ((batch + 1) * 4)
