import os
import torch
from torchvision.transforms import functional as F
import numpy as np
from util import Adder
from data import test_dataloader
import torch.nn as nn
import cv2

def _eval(model, args):
    kernel_size = args.kernel_size
    temp = np.zeros((2*kernel_size+1, 2*kernel_size+1))
    temp[kernel_size, kernel_size] = 1
    filter_temp = cv2.GaussianBlur(temp, (kernel_size, kernel_size), 0, 0)
    constant = np.max(filter_temp)
    state_dict = torch.load(args.test_model)
    model.load_state_dict(state_dict['model'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = test_dataloader(args.data_dir, args.kernel_size, batch_size=1, num_workers=0, use_transform=True)
    torch.cuda.empty_cache()
    model.eval()
    with torch.no_grad():
        mse_adder = Adder()
        for i in range(50):
            for iter_idx, data in enumerate(dataloader):
                input_img, label_img, name = data
                input_img = input_img.to(device)
                pred = model(input_img)
                pred_clip = torch.clamp(pred, 0, 1)
                pred_clip = pred_clip * constant

                label_img = label_img/torch.max(label_img)
                pred_clip = pred_clip/torch.max(pred_clip)

                if args.save_image:
                    save_name = os.path.join(args.result_dir, name[0])
                    # pred_clip += 0.5 / 255
                    pre = F.to_pil_image(pred_clip[0, 0, :, :].detach().cpu())
                    label = F.to_pil_image(label_img[0, 0, :, :])
                    pre.save(save_name)


        print('==========================================================')
        print('The average MSE is %.6f' % (mse_adder.average()))
