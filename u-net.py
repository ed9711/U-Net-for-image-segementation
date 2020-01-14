from __future__ import print_function
import torch
from data import *
from model1 import *
from train import *
from data_aug import *
import numpy as np
import logging
import torch.nn as nn

if __name__ == "__main__":
    np.random.seed(0)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')


    # masks_path = glob.glob(str('./cat_data/train/mask') + "/*")
    # images_path = glob.glob(str('./cat_data/train/input') + str("/*"))
    # print("Augumenting {} images".format(len(images_path)))
    # for i in range(4):
    #     data_aug('./cat_data/train/input', './cat_data/train/mask', images_path, masks_path, i)

    # Dataset
    trainData = GetData('./cat_data/train/input', './cat_data/train/mask')
    testData = GetData('./cat_data/test/input', './cat_data/test/mask')
    # trainData1 = GetData('./cat_data/train/input', './cat_data/train/mask')

    # Dataloader
    trainLoad = torch.utils.data.DataLoader(dataset=trainData, num_workers=6, batch_size=1, shuffle=True)
    testLoad = torch.utils.data.DataLoader(dataset=testData, num_workers=2, batch_size=1, shuffle=True)
    # trainLoad1 = torch.utils.data.DataLoader(dataset=trainData1, num_workers=6, batch_size=1, shuffle=True)

    num_colours = 3
    num_in_channels = 3
    kernel = 3
    num_filters = 32
    cnn = UNet1(kernel, num_filters, num_colours, num_in_channels)
    cnn.to(device=device)

    logging.info(f'Network:\n'
                 f'\t{num_in_channels} input channels\n'
                 f'\t{num_colours} output channels\n')

    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)
    result = train_net(trainLoad, cnn, optimizer, name=0, epoch=20, device=device)
    result1 = test_model(result[1], testLoad, i=0, epoch=20, make_prediction=True, save_folder_name='result')
    print("test loss: {}, test acc: {}".format(result1[1], result1[0]))

    # cnn1 = UNet1(kernel, num_filters, num_colours, num_in_channels)
    # cnn1.to(device=device)
    # optimizer1 = torch.optim.Adam(cnn1.parameters(), lr=0.001)
    # result = train_net(trainLoad, cnn1, optimizer1, name=1, epoch=20, device=device)
    # result1 = test_model(result[1], testLoad, i=1, epoch=20, make_prediction=True, save_folder_name='result')
    # print("test loss: {}, test acc: {}".format(result1[1], result1[0]))


    # # reinitialize and reshaping the finalconv
    # # https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html#initialize-and-reshape-the-networks
    # cnn.finalconv = nn.Conv2d(6, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # # this is for finetuneing, so all the params are to be updated
    # params_to_update = cnn.parameters()
    # for name, param in cnn.named_parameters():
    #     if param.requires_grad == True:
    #         print("\t", name)
    # optimizer = torch.optim.Adam(params_to_update, lr=0.001)
    # result = train_net(trainLoad1, cnn, optimizer, name=0, epoch=20, device=device)
    # result1 = test_model("outputs1/model_lossF_CEL.pwf", testLoad, i=1, epoch=20, make_prediction=True, save_folder_name='result')
    # print("test loss: {}, test acc: {}".format(result1[1], result1[0]))




