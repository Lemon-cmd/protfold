import sys, time, argparse, matplotlib.pyplot as plt
import torch, einops, sidechainnet as scn, numpy as np
import torch.nn as nn, torch.optim as optim, torch.nn.functional as F

from math import pi
from tqdm.notebook import tqdm
from collections import defaultdict

from einops import rearrange
from einops.layers.torch import Rearrange
from torch.distributions import VonMises

from vit import MAE
from resnet import ResNet
from utils import get_seq_features, CropsDataset, train

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def parse_args():
    parser = argparse.ArgumentParser(description='train_script.py')

    parser.add_argument('-t', dest='model_type', default='cnn', type=str)

    # checkpoint file
    parser.add_argument('-cfname', dest='ckpt_fname', default="", type=str)

    parser.add_argument('-e', dest='epochs', default=10, type=int)

    parser.add_argument('-c', dest='crop_sz', default=64, type=int)

    parser.add_argument('-s', dest='stride_sz', default=48, type=int)

    parser.add_argument('-p', dest='pad_sz', default=1, type=int)

    parser.add_argument('-b', dest='batch_size', default=32, type=int)

    parser.add_argument('-casp', dest='casp_version', default=12, type=int)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    torch.cuda.empty_cache()

    data = scn.load(scn_dir = "../data/", casp_version=args.casp_version, with_pytorch="dataloaders",
                seq_as_onehot=True, aggregate_model_input=False, batch_size=16, dynamic_batching=True)

    if args.model_type == 'cnn':
        # resnet
        model = ResNet(41, 64, 1, num_blks=20, dropout=0.1)
        
    elif args.model_type == 'vit':
        # vit / mae model
        model = MAE(
            in_channels=41, out_channels=64, encoder_dim=64, decoder_dim=64,
            encoder_mlp_dim=1, encoder_depth=1, encoder_heads=1, encoder_dim_head=32, encoder_dropout=0.5,
            decoder_mlp_dim=1, decoder_depth=1, decoder_heads=1, decoder_dim_head=32, decoder_dropout=0.5,
            patch_size=16, pos_max_len=1000
        )

    else:
        print("Model type not found")

    print("Total no. of params:", get_n_params(model))

    fname = args.ckpt_fname
    batch_sz = args.batch_size * torch.cuda.device_count()

    num_dist_bins=64
    model, optimizer = train(model, data, args.crop_sz, args.stride_sz, args.pad_sz, num_dist_bins, batch_sz, fname, args.epochs)

if __name__ == "__main__":
    main()
