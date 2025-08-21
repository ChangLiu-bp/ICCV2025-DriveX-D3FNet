import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V

import cv2
import os
import numpy as np

from time import time
from tqdm import tqdm  # 导入 tqdm

from networks.unet import Unet
from networks.dunet import Dunet
from networks.dinknet import LinkNet34, DinkNet34, DinkNet50, DinkNet101, DinkNet34_less_pool
from framework import MyFrame
from loss import dice_bce_loss
from data import ImageFolder

def main():
    SHAPE = (1024, 1024)
    #path of train dateset
    ROOT = ' '
    imagelist = filter(lambda x: x.find('sat') != -1, os.listdir(ROOT))
    trainlist = map(lambda x: x[:-8], imagelist)
    NAME = ' '
    BATCHSIZE_PER_CARD = 8

    solver = MyFrame(DinkNet34, dice_bce_loss, 2e-4)
    batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD

    dataset = ImageFolder(trainlist, ROOT)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=4
    )

    mylog = open('logs/' + NAME + '.log', 'w')
    tic = time()
    no_optim = 0
    total_epoch = 150
    train_epoch_best_loss = 100.

    for epoch in range(1, total_epoch + 1):
        data_loader_iter = iter(data_loader)
        train_epoch_loss = 0
        for img, mask in tqdm(data_loader_iter, desc=f'Epoch {epoch}', unit='batch'):
            solver.set_input(img, mask)
            train_loss = solver.optimize()
            train_epoch_loss += train_loss
        train_epoch_loss /= len(data_loader_iter)
        
        print('********', file=mylog)
        print(f'epoch: {epoch}    time: {int(time() - tic)}', file=mylog)
        print(f'train_loss: {train_epoch_loss}', file=mylog)
        print(f'SHAPE: {SHAPE}', file=mylog)
        print('********')
        print(f'epoch: {epoch}    time: {int(time() - tic)}')
        print(f'train_loss: {train_epoch_loss}')
        print(f'SHAPE: {SHAPE}')
        
        if train_epoch_loss >= train_epoch_best_loss:
            no_optim += 1
        else:
            no_optim = 0
            train_epoch_best_loss = train_epoch_loss
            solver.save('weights/' + NAME + '.pth')
        
        if no_optim > 6:
            print(f'early stop at {epoch} epoch', file=mylog)
            print(f'early stop at {epoch} epoch')
            break
        
        if no_optim > 3:
            if solver.old_lr < 5e-7:
                break
            solver.load('weights/' + NAME + '.pth')
            solver.update_lr(5.0, factor=True, mylog=mylog)
        
        mylog.flush()

    print('Finish!', file=mylog)
    print('Finish!')
    mylog.close()

if __name__ == '__main__':
    main()
