import torch
from torchvision.transforms import functional as F
from data import valid_dataloader
from util import Adder
import os
from skimage.metrics import peak_signal_noise_ratio
import torch.nn as nn


def _valid(model, args, ep):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gopro = valid_dataloader(args.data_dir, args.kernel_size, batch_size=1, num_workers=0)
    model.eval()
    mse_adder = Adder()

    with torch.no_grad():
        print('Start GoPro Evaluation')
        for idx, data in enumerate(gopro):
            input_img, label_img = data
            input_img = input_img.to(device)
            label_img = label_img.to(device)

            pred = model(input_img)
            pred_clip = torch.clamp(pred, 0, 1)

            eva = nn.MSELoss()
            mse = eva(pred_clip, label_img)
            mse_adder(mse)

            print('\r%03d'%idx, end=' ')

    print('\n')
    model.train()
    return mse_adder.average()
