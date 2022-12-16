import os
import torch
import argparse
from torch.backends import cudnn
from train import _train
from eval import _eval
from model import UNet


def main(args):
    # CUDNN
    cudnn.benchmark = True

    if not os.path.exists('results/'):
        os.makedirs(args.model_save_dir)
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    model = UNet(n_channels=3)
    if torch.cuda.is_available():
        model.cuda()
    if args.mode == 'train':
        _train(model, args)

    elif args.mode == 'test':
        _eval(model, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data for train
    parser.add_argument('--data_dir', type=str, default=r'dataset\ChildHospData40x')
    parser.add_argument('--mode', default='train', choices=['train', 'test'], type=str)
    # Train
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--num_epoch', type=int, default=500)
    parser.add_argument('--print_freq', type=int, default=20)
    parser.add_argument('--num_worker', type=int, default=8)
    parser.add_argument('--save_freq', type=int, default=50)
    parser.add_argument('--valid_freq', type=int, default=20)
    parser.add_argument('--resume', type=str, default=r'')
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--lr_steps', type=list, default=[(x+1) * 50 for x in range(500//50)])

    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--patch_factor', type=int, default=1)
    parser.add_argument('--ave_spectrum', action='store_true', help='whether to use minibatch average spectrum')
    parser.add_argument('--log_matrix', action='store_true', help='whether to adjust the spectrum weight matrix by logarithm')
    parser.add_argument('--batch_matrix', action='store_true', help='whether to calculate the spectrum weight matrix using batch-based statistics')
    parser.add_argument('--freq_start_epoch', type=int, default=1, help='the start epoch to add focal frequency loss')
    parser.add_argument('-kernel_size', type=int, default=25, help='the diameter of the object')
    # Test
    parser.add_argument('--test_model', type=str, default='results/weights/Best.pkl')
    parser.add_argument('--save_image', type=bool, default=True, choices=[True, False])

    args = parser.parse_args()
    args.model_save_dir = r'results/weights/'
    args.result_dir = r'results/images/6/'
    print(args)
    main(args)
