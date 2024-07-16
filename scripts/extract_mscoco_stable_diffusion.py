import torch
import os
import numpy as np
import sys
sys.path.insert(0,'/home/min/a/long273/Documents/diffusers/U-ViT')
import libs.autoencoder
import libs.clip
from datasets import MSCOCODatabase
import argparse
from tqdm import tqdm
import json

from diffusers import AutoencoderKL
#TODO: change the resolution from 256 to 512 to generate 64x64 latents
def main(resolution=256):
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='train')
    args = parser.parse_args()
    print(args)


    if args.split == "train":
        datas = MSCOCODatabase(root='/home/nano01/a/long273/train2017',#change to 2017 from 2014
                             annFile='/home/nano01/a/long273/annotations/captions_train2017.json',
                             size=resolution)
        save_dir = f'/home/nano01/a/long273/coco{resolution}_features/trainSD'
    elif args.split == "val":
        datas = MSCOCODatabase(root='/home/nano01/a/long273/val2017',
                             annFile='/home/nano01/a/long273/annotations/captions_val2017.json',
                             size=resolution)
        save_dir = f'/home/nano01/a/long273/coco{resolution}_features/valSD'
    else:
        raise NotImplementedError("ERROR!")

    device = "cuda:0"
    #os.makedirs(save_dir)
    use_category_id = True


    autoencoder = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae", use_safetensors=True)        
    #autoencoder = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-1", subfolder="vae", use_safetensors=True)        
    #autoencoder = libs.autoencoder.get_model('assets/stable-diffusion/autoencoder_kl.pth')
    autoencoder.to(device)     
    
    clip = libs.clip.FrozenCLIPEmbedder()
    clip.eval()
    clip.to(device)

    with torch.no_grad():
        for idx, data in tqdm(enumerate(datas)):
            x, captions, segmentation = data

            #print('save captions')
            
            #np.savetxt(os.path.join(save_dir, f'{idx}_text.txt'), captions, delimiter=" ", fmt="%s") 
            
            if len(x.shape) == 3:
                x = x[None, ...]
            x = torch.tensor(x, device=device)

            #print('x shape',x.shape) #1,3,256,256
            moments=autoencoder.encode(x, return_dict=False)[1].squeeze(0)
            #moments = autoencoder(x, fn='encode_moments').squeeze(0)
            #print('moments',moments.shape) #8,32,32
            moments = moments.detach().cpu().numpy()
            np.save(os.path.join(save_dir, f'{idx}.npy'), moments)
            
            latent = clip.encode(captions)
            for i in range(len(latent)):
                c = latent[i].detach().cpu().numpy()
                np.save(os.path.join(save_dir, f'{idx}_{i}.npy'), c)
            
            #TODO: save panoptic annotation
            #print("save annotion,", idx)
            if use_category_id==True:
                np.save(os.path.join(save_dir, f'{idx}_seg.npy'), segmentation)
            else:
                if len(segmentation.shape) == 3:
                    segmentation = segmentation[None, ...]
                segmentation = torch.tensor(segmentation, device=device)
                encode_maps = autoencoder(segmentation).squeeze(0)
                encode_maps = encode_maps.detach().cpu().numpy()
                np.save(os.path.join(save_dir, f'{idx}_encode_p.npy'), encode_maps)
            
if __name__ == '__main__':
    main()
