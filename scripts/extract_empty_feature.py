import torch
import os
import numpy as np
import sys
sys.path.insert(0,'/home/min/a/long273/Documents/diffusers/U-ViT')
print(sys.path)
import libs.autoencoder
import libs.clip
from datasets import MSCOCODatabase
import argparse
from tqdm import tqdm


def main():
    prompts = [
        '',
    ]

    device = 'cuda'
    clip = libs.clip.FrozenCLIPEmbedder()
    clip.eval()
    clip.to(device)

    save_dir = f'/home/nano01/a/long273/coco256_features'
    latent = clip.encode(prompts)
    print(latent.shape)
    c = latent[0].detach().cpu().numpy()
    np.save(os.path.join(save_dir, f'empty_context.npy'), c)


if __name__ == '__main__':
    main()
