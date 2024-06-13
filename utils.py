import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
from torchvision.utils import save_image, make_grid, draw_segmentation_masks
from absl import logging
import wandb
from panopticapi.utils import IdGenerator
import json
from PIL import Image


def set_logger(log_level='info', fname=None):
    import logging as _logging
    handler = logging.get_absl_handler()
    formatter = _logging.Formatter('%(asctime)s - %(filename)s - %(message)s')
    handler.setFormatter(formatter)
    logging.set_verbosity(log_level)
    if fname is not None:
        handler = _logging.FileHandler(fname)
        handler.setFormatter(formatter)
        logging.get_absl_logger().addHandler(handler)


def dct2str(dct):
    return str({k: f'{v:.6g}' for k, v in dct.items()})


def get_nnet(name, **kwargs):
    if name == 'uvit':
        from libs.uvit import UViT
        return UViT(**kwargs)
    elif name == 'uvit_t2i':
        from libs.uvit_t2i import UViT
        return UViT(**kwargs)
    else:
        raise NotImplementedError(name)


def set_seed(seed: int):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)


def get_optimizer(params, name, **kwargs):
    if name == 'adam':
        from torch.optim import Adam
        return Adam(params, **kwargs)
    elif name == 'adamw':
        from torch.optim import AdamW
        return AdamW(params, **kwargs)
    else:
        raise NotImplementedError(name)


def customized_lr_scheduler(optimizer, warmup_steps=-1):
    from torch.optim.lr_scheduler import LambdaLR
    def fn(step):
        if warmup_steps > 0:
            return min(step / warmup_steps, 1)
        else:
            return 1
    return LambdaLR(optimizer, fn)


def get_lr_scheduler(optimizer, name, **kwargs):
    if name == 'customized':
        return customized_lr_scheduler(optimizer, **kwargs)
    elif name == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, **kwargs)
    else:
        raise NotImplementedError(name)


def ema(model_dest: nn.Module, model_src: nn.Module, rate):
    param_dict_src = dict(model_src.named_parameters())
    for p_name, p_dest in model_dest.named_parameters():
        p_src = param_dict_src[p_name]
        assert p_src is not p_dest
        p_dest.data.mul_(rate).add_((1 - rate) * p_src.data)


class TrainState(object):
    def __init__(self, optimizer, lr_scheduler, step, nnet=None, nnet_ema=None):
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.step = step
        self.nnet = nnet
        self.nnet_ema = nnet_ema

    def set_nnet(self, nnet):#NOTE:customize nnet 
        self.nnet = nnet
        self.nnet_ema= nnet
        self.ema_update(0)
        self.to(device)
        
    def ema_update(self, rate=0.9999):
        if self.nnet_ema is not None:
            ema(self.nnet_ema, self.nnet, rate)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.step, os.path.join(path, 'step.pth'))
        for key, val in self.__dict__.items():
            if key != 'step' and val is not None:
                torch.save(val.state_dict(), os.path.join(path, f'{key}.pth'))

    def load(self, path):
        logging.info(f'load from {path}')
        self.step = torch.load(os.path.join(path, 'step.pth'))
        for key, val in self.__dict__.items():
            if key != 'step' and val is not None:
                val.load_state_dict(torch.load(os.path.join(path, f'{key}.pth'), map_location='cpu'))

    def resume(self, ckpt_root, step=None):
        if not os.path.exists(ckpt_root):
            return
        if step is None:
            ckpts = list(filter(lambda x: '.ckpt' in x, os.listdir(ckpt_root)))
            if not ckpts:
                return
            if not ckpts[0].split(".")[0].isnumeric():
                ckpt_path = os.path.join(ckpt_root, f'best.ckpt')
                logging.info(f'resume from {ckpt_path}')
                self.load(ckpt_path)
                return
            else:
                steps = map(lambda x: int(x.split(".")[0]), ckpts)
                step = max(steps)
        ckpt_path = os.path.join(ckpt_root, f'{step}.ckpt')
        logging.info(f'resume from {ckpt_path}')
        self.load(ckpt_path)

    def to(self, device):
        for key, val in self.__dict__.items():
            if isinstance(val, nn.Module):
                val.to(device)


def cnt_params(model):
    return sum(param.numel() for param in model.parameters())


def initialize_train_state(config, device):
    params = []

    nnet = get_nnet(**config.nnet)
    params += nnet.parameters()
    nnet_ema = get_nnet(**config.nnet)
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    train_state = TrainState(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0,
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state.ema_update(0)
    train_state.to(device)
    return train_state


def amortize(n_samples, batch_size):
    k = n_samples // batch_size
    r = n_samples % batch_size
    return k * [batch_size] if r == 0 else k * [batch_size] + [r]

def category2rgb(color_generator,id_map, categegories):
    
    rgb_shape = tuple([3]+list(id_map.shape))
    rgb_map = torch.zeros(rgb_shape, dtype=torch.uint8)
    for i in range(id_map.shape[0]):
        for j in range(id_map.shape[1]):
            c=id_map[i,j].item()
            while c not in categegories:
                c-=1
                if c==0:
                    c=1
                    break
            rgb_map[:,i,j] = torch.tensor(color_generator.get_color(c))
        
    return rgb_map

#NOTE: analog bits for generating panoptic segmentation masks from bit diffusion paper
#TODO: rewrite with pytorch
def int2bits(x, n=8, out_dtype=None):
  """Convert an integer x in (b,1,h,w) into bits in (b,n,h,w)."""
  #print(x.shape, x[0,:,0,0])
  x=x.type(torch.int)
  y=x.clone()
  for i in range(1,n):
      y = torch.cat((torch.bitwise_right_shift(x, i), y), dim=1)
  #print('int2bits:',y.shape, y[0,:,0,0])
  y = torch.remainder(y, 2)
  #print('mod by 2:',y.shape, y[0,:,0,0])
  if out_dtype and out_dtype != y.dtype:
    y = y.type(out_dtype)
  return y

def bits2int(x, out_dtype=torch.int, n=8):
  """Converts bits x in (b,n,h,w) into an integer in (b,1,h,w)."""
  x = x.type(out_dtype)
  y = torch.zeros(x.shape[0],x.shape[2],x.shape[3],device=x.device)
  for i in range(n):
    y += x[:,i,:,:] * (2 ** i)
 
  #x = torch.sum(x * (2 ** torch.range(start=0,end=x.shape[1])), 1, keepdim=True)
  return y.unsqueeze(1)

#NOTE: category ID = R * 256 * G + 256 * 256 + B.
colormap= torch.randint(0,255,(256,3)).cuda()
def color_map(x):
    #input B1HW, reshape to BHW
    x=x.squeeze(1)
    #then map to BHW3
    x=  colormap[x.type(torch.long)]
    #reshape back to B3HW
    x=x.permute(0,3,1,2)
    #print(x.shape)
    return x

def mse(a):  # mean of square
    b=a.clone().float()
    return b.pow(2).mean(dim=-1)
def eval_mask_cnt(pred_mask, panoptic):#compare the number of category pixels in generated mask and original ones
    batch= pred_mask.shape[0]
    pd= pred_mask.flatten(start_dim=1)
    gt= panoptic.flatten(start_dim=1)
    cnt_diff=0.
    for i in range(batch):
        pred_cnt = torch.bincount(pd[i,:].int(), minlength=201)
        gt_cnt = torch.bincount(gt[i,:].int(), minlength=201)
        cnt_diff+= mse(pred_cnt[:201]- gt_cnt[:201])

    return cnt_diff/batch
def sample2dir(accelerator, path, n_samples, mini_batch_size, sample_fn, unpreprocess_fn=None, step=None, use_panoptic=False, mask_path=None):
    os.makedirs(path, exist_ok=True)
    idx = 0
    batch_size = mini_batch_size * accelerator.num_processes
    loss_mask_all= []
    eval_cnt_mask_diff= []
    panoptic_coco_categories = '../panopticapi-master/panoptic_coco_categories.json'

    with open(panoptic_coco_categories, 'r') as f:
        categories_list = json.load(f)
    categegories = {category['id']: category for category in categories_list}

    color_generator=IdGenerator(categegories)
    print_text=False
    for _batch_size in tqdm(amortize(n_samples, batch_size), disable=not accelerator.is_main_process, desc='sample2dir'):
        if idx==0:
            print_text=True
        else:
            print_text=False
        if use_panoptic==False:
            samples = sample_fn(mini_batch_size, use_panoptic=False, print_text=print_text)
        else:
            sample_idx, samples, pred_mask, loss_mask, panoptic = sample_fn(mini_batch_size,use_panoptic=True, print_text=print_text)
            #TODO:accumulate loss
            loss_mask_all.append(loss_mask)
            pred_mask = accelerator.gather(pred_mask.contiguous())[:_batch_size]
        samples = unpreprocess_fn(samples)
        samples = accelerator.gather(samples.contiguous())[:_batch_size]
        if accelerator.is_main_process:
            if idx==0 and (step is not None):
                grid_samples = make_grid(samples, 8)
                wandb.log({'eval_samples': wandb.Image(grid_samples)}, step)
            #TODO: convert analog bits back to integer
            if use_panoptic==True:
                pred_mask= bits2int(pred_mask>0, torch.int) #this convert pred_mask to [N,1,H,W]
                #Evaluate masks by comparing bin counts
                cnt_diff= eval_mask_cnt(pred_mask, panoptic)
                eval_cnt_mask_diff.append(cnt_diff)

                color_masks= color_map(pred_mask)
            for i,sample in enumerate(samples):
                #img = Image.fromarray(color_masks[i,:,:,:], 'RGB')
                #img.save(os.path.join(mask_path, f"{sample_idx[i]}.png"))
                #save_image(color_masks, os.path.join(mask_path, f"{sample_idx[i]}.png"))
                save_image(sample, os.path.join(path, f"{sample_idx[i]}.png"))
                idx += 1
            if idx==0 and (step is not None) and use_panoptic==True: #visualize in wandb
                
                #color_mask = torch.zeros_like(pred_mask)
                #for i in range(mask_label.shape[0]):
                #    color_mask[i,...] = category2rgb(color_generator,mask_label[i,...],categegories)
                
                grid_mask = make_grid(color_masks.float() , 8) 
                '''
                #TODO: print colored maps
                mask_max, mask_label = torch.max(pred_mask,dim=1, keepdim=True)#indices
                color_masks = torch.zeros_like(pred_mask, dtype=bool) #[b,200,h,w]
                color_masks[pred_mask==mask_max] = 1
                empty_image = torch.zeros(pred_mask.shape[0],3,32,32, dtype=torch.uint8)
                for i in range(pred_mask.shape[0]):
                    logging.info(f'Color masks...{i}')
                    empty_image[i,:,:,:]= draw_segmentation_masks(empty_image[i,:,:,:], color_masks[i,:,:,:],alpha=0.7)   
                grid_mask = make_grid(empty_image.float() , 8, normalize=True) 
                '''
                #grid_mask = make_grid(mask_label.float(), 8, normalize=True)
                wandb.log({'pred_mask': wandb.Image(grid_mask)}, step)

                color_panoptic= color_map(panoptic)
                ground_mask = make_grid(color_panoptic.float(), 8)
                wandb.log({'ground_truth_mask': wandb.Image(ground_mask)}, step)
           
    if use_panoptic==True and (step is not None):
        wandb.log({f'eval_loss_mask': torch.mean(torch.stack(loss_mask_all))}, step)
        wandb.log({f'eval_cnt_mask_diff': torch.mean(torch.stack(eval_cnt_mask_diff))}, step)


def grad_norm(model):
    total_norm = 0.
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm
