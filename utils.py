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
import math


def create_unet_diffusers_config(original_config):
    """
    Creates a config for the diffusers based on the config of the LDM model.
    """
    unet_params = original_config.model.params.unet_config.params

    block_out_channels = [unet_params.model_channels * mult for mult in unet_params.channel_mult]

    down_block_types = []
    resolution = 1
    for i in range(len(block_out_channels)):
        block_type = "CrossAttnDownBlock2D" if resolution in unet_params.attention_resolutions else "DownBlock2D"
        down_block_types.append(block_type)
        if i != len(block_out_channels) - 1:
            resolution *= 2

    up_block_types = []
    for i in range(len(block_out_channels)):
        block_type = "CrossAttnUpBlock2D" if resolution in unet_params.attention_resolutions else "UpBlock2D"
        up_block_types.append(block_type)
        resolution //= 2

    config = dict(
        sample_size=64,
        in_channels=unet_params.in_channels,
        out_channels=unet_params.out_channels,
        down_block_types=tuple(down_block_types),
        up_block_types=tuple(up_block_types),
        block_out_channels=tuple(block_out_channels),
        layers_per_block=unet_params.num_res_blocks,
        cross_attention_dim=unet_params.context_dim,
        attention_head_dim=unet_params.num_heads,
    )

    return config

def shave_segments(path, n_shave_prefix_segments=1):
    """
    Removes segments. Positive values shave the first segments, negative shave the last segments.
    """
    if n_shave_prefix_segments >= 0:
        return '.'.join(path.split('.')[n_shave_prefix_segments:])
    else:
        return '.'.join(path.split('.')[:n_shave_prefix_segments])

def renew_resnet_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside resnets to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item.replace('in_layers.0', 'norm1')
        new_item = new_item.replace('in_layers.2', 'conv1')

        new_item = new_item.replace('out_layers.0', 'norm2')
        new_item = new_item.replace('out_layers.3', 'conv2')

        new_item = new_item.replace('emb_layers.1', 'time_emb_proj')
        new_item = new_item.replace('skip_connection', 'conv_shortcut')

        new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

        mapping.append({'old': old_item, 'new': new_item})

    return mapping

def renew_attention_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside attentions to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item

        # new_item = new_item.replace('norm.weight', 'group_norm.weight')
        # new_item = new_item.replace('norm.bias', 'group_norm.bias')

        # new_item = new_item.replace('proj_out.weight', 'proj_attn.weight')
        # new_item = new_item.replace('proj_out.bias', 'proj_attn.bias')

        # new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

        mapping.append({'old': old_item, 'new': new_item})

    return mapping
def assign_to_checkpoint(paths, checkpoint, old_checkpoint, attention_paths_to_split=None, additional_replacements=None, config=None):
    """
    This does the final conversion step: take locally converted weights and apply a global renaming
    to them. It splits attention layers, and takes into account additional replacements
    that may arise.

    Assigns the weights to the new checkpoint.
    """
    assert isinstance(paths, list), "Paths should be a list of dicts containing 'old' and 'new' keys."

    # Splits the attention layers into three variables.
    if attention_paths_to_split is not None:
        for path, path_map in attention_paths_to_split.items():
            old_tensor = old_checkpoint[path]
            channels = old_tensor.shape[0] // 3

            target_shape = (-1, channels) if len(old_tensor.shape) == 3 else (-1)

            num_heads = old_tensor.shape[0] // config["num_head_channels"] // 3

            old_tensor = old_tensor.reshape((num_heads, 3 * channels // num_heads) + old_tensor.shape[1:])
            query, key, value = old_tensor.split(channels // num_heads, dim=1)

            checkpoint[path_map['query']] = query.reshape(target_shape)
            checkpoint[path_map['key']] = key.reshape(target_shape)
            checkpoint[path_map['value']] = value.reshape(target_shape)

    for path in paths:
        new_path = path['new']

        # These have already been assigned
        if attention_paths_to_split is not None and new_path in attention_paths_to_split:
            continue

        # Global renaming happens here
        new_path = new_path.replace('middle_block.0', 'mid_block.resnets.0')
        new_path = new_path.replace('middle_block.1', 'mid_block.attentions.0')
        new_path = new_path.replace('middle_block.2', 'mid_block.resnets.1')

        if additional_replacements is not None:
            for replacement in additional_replacements:
                new_path = new_path.replace(replacement['old'], replacement['new'])

        # proj_attn.weight has to be converted from conv 1D to linear
        if "proj_attn.weight" in new_path:
            checkpoint[new_path] = old_checkpoint[path['old']][:, :, 0]
        else:
            checkpoint[new_path] = old_checkpoint[path['old']]

def convert_ldm_unet_checkpoint(checkpoint, config):
    """
    Takes a state dict and a config, and returns a converted checkpoint.
    """

    # extract state_dict for UNet
    unet_state_dict = {}
    unet_key = "model.diffusion_model."
    keys = list(checkpoint.keys())
    for key in keys:
        if key.startswith(unet_key):
            unet_state_dict[key.replace(unet_key, "")] = checkpoint.pop(key)

    new_checkpoint = {}

    new_checkpoint['time_embedding.linear_1.weight'] = unet_state_dict['time_embed.0.weight']
    new_checkpoint['time_embedding.linear_1.bias'] = unet_state_dict['time_embed.0.bias']
    new_checkpoint['time_embedding.linear_2.weight'] = unet_state_dict['time_embed.2.weight']
    new_checkpoint['time_embedding.linear_2.bias'] = unet_state_dict['time_embed.2.bias']

    new_checkpoint['conv_in.weight'] = unet_state_dict['input_blocks.0.0.weight']
    new_checkpoint['conv_in.bias'] = unet_state_dict['input_blocks.0.0.bias']

    new_checkpoint['conv_norm_out.weight'] = unet_state_dict['out.0.weight']
    new_checkpoint['conv_norm_out.bias'] = unet_state_dict['out.0.bias']
    new_checkpoint['conv_out.weight'] = unet_state_dict['out.2.weight']
    new_checkpoint['conv_out.bias'] = unet_state_dict['out.2.bias']

    # Retrieves the keys for the input blocks only
    num_input_blocks = len({'.'.join(layer.split('.')[:2]) for layer in unet_state_dict if 'input_blocks' in layer})
    input_blocks = {layer_id: [key for key in unet_state_dict if f'input_blocks.{layer_id}' in key] for layer_id in range(num_input_blocks)}

    # Retrieves the keys for the middle blocks only
    num_middle_blocks = len({'.'.join(layer.split('.')[:2]) for layer in unet_state_dict if 'middle_block' in layer})
    middle_blocks = {layer_id: [key for key in unet_state_dict if f'middle_block.{layer_id}' in key] for layer_id in range(num_middle_blocks)}

    # Retrieves the keys for the output blocks only
    num_output_blocks = len({'.'.join(layer.split('.')[:2]) for layer in unet_state_dict if 'output_blocks' in layer})
    output_blocks = {layer_id: [key for key in unet_state_dict if f'output_blocks.{layer_id}' in key] for layer_id in range(num_output_blocks)}

    for i in range(1, num_input_blocks):
        block_id = (i - 1) // (config['layers_per_block'] + 1)
        layer_in_block_id = (i - 1) % (config['layers_per_block'] + 1)

        resnets = [key for key in input_blocks[i] if f'input_blocks.{i}.0' in key and f'input_blocks.{i}.0.op' not in key]
        attentions = [key for key in input_blocks[i] if f'input_blocks.{i}.1' in key]

        if f'input_blocks.{i}.0.op.weight' in unet_state_dict:
            new_checkpoint[f'down_blocks.{block_id}.downsamplers.0.conv.weight'] = unet_state_dict.pop(f'input_blocks.{i}.0.op.weight')
            new_checkpoint[f'down_blocks.{block_id}.downsamplers.0.conv.bias'] = unet_state_dict.pop(f'input_blocks.{i}.0.op.bias')

        paths = renew_resnet_paths(resnets)
        meta_path = {'old': f'input_blocks.{i}.0', 'new': f'down_blocks.{block_id}.resnets.{layer_in_block_id}'}
        assign_to_checkpoint(paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config)

        if len(attentions):
            paths = renew_attention_paths(attentions)
            meta_path = {'old': f'input_blocks.{i}.1', 'new': f'down_blocks.{block_id}.attentions.{layer_in_block_id}'}
            assign_to_checkpoint(paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config)


    resnet_0 = middle_blocks[0]
    attentions = middle_blocks[1]
    resnet_1 = middle_blocks[2]

    resnet_0_paths = renew_resnet_paths(resnet_0)
    assign_to_checkpoint(resnet_0_paths, new_checkpoint, unet_state_dict, config=config)

    resnet_1_paths = renew_resnet_paths(resnet_1)
    assign_to_checkpoint(resnet_1_paths, new_checkpoint, unet_state_dict, config=config)

    attentions_paths = renew_attention_paths(attentions)
    meta_path = {'old': 'middle_block.1', 'new': 'mid_block.attentions.0'}
    assign_to_checkpoint(attentions_paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config)

    for i in range(num_output_blocks):
        block_id = i // (config['layers_per_block'] + 1)
        layer_in_block_id = i % (config['layers_per_block'] + 1)
        output_block_layers = [shave_segments(name, 2) for name in output_blocks[i]]
        output_block_list = {}

        for layer in output_block_layers:
            layer_id, layer_name = layer.split('.')[0], shave_segments(layer, 1)
            if layer_id in output_block_list:
                output_block_list[layer_id].append(layer_name)
            else:
                output_block_list[layer_id] = [layer_name]

        if len(output_block_list) > 1:
            resnets = [key for key in output_blocks[i] if f'output_blocks.{i}.0' in key]
            attentions = [key for key in output_blocks[i] if f'output_blocks.{i}.1' in key]

            resnet_0_paths = renew_resnet_paths(resnets)
            paths = renew_resnet_paths(resnets)

            meta_path = {'old': f'output_blocks.{i}.0', 'new': f'up_blocks.{block_id}.resnets.{layer_in_block_id}'}
            assign_to_checkpoint(paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config)

            if ['conv.weight', 'conv.bias'] in output_block_list.values():
                index = list(output_block_list.values()).index(['conv.weight', 'conv.bias'])
                new_checkpoint[f'up_blocks.{block_id}.upsamplers.0.conv.weight'] = unet_state_dict[f'output_blocks.{i}.{index}.conv.weight']
                new_checkpoint[f'up_blocks.{block_id}.upsamplers.0.conv.bias'] = unet_state_dict[f'output_blocks.{i}.{index}.conv.bias']

                # Clear attentions as they have been attributed above.
                if len(attentions) == 2:
                    attentions = []

            if len(attentions):
                paths = renew_attention_paths(attentions)
                meta_path = {
                    'old': f'output_blocks.{i}.1',
                    'new': f'up_blocks.{block_id}.attentions.{layer_in_block_id}'
                }
                assign_to_checkpoint(paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config)
        else:
            resnet_0_paths = renew_resnet_paths(output_block_layers, n_shave_prefix_segments=1)
            for path in resnet_0_paths:
                old_path = '.'.join(['output_blocks', str(i), path['old']])
                new_path = '.'.join(['up_blocks', str(block_id), 'resnets', str(layer_in_block_id), path['new']])

                new_checkpoint[new_path] = unet_state_dict[old_path]

    return new_checkpoint



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
        #print(p_name, p_src, p_dest)
        #assert p_src is not p_dest
        p_dest.data.mul_(rate).add_((1 - rate) * p_src.data)


class TrainState(object):
    def __init__(self, optimizer, lr_scheduler, step, nnet=None, nnet_ema=None):
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.step = step
        self.nnet = nnet
        self.nnet_ema = nnet_ema

    '''
    def set_nnet(self, unet):#NOTE:customize nnet 
        self.nnet = unet
        self.nnet_ema= unet
        self.ema_update(0)
        self.to(device)
    '''
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
            return False
        if step is None:
            ckpts = list(filter(lambda x: '.ckpt' in x, os.listdir(ckpt_root)))
            if not ckpts:
                return False
            if not ckpts[0].split(".")[0].isnumeric():
                ckpt_path = os.path.join(ckpt_root, f'best.ckpt')
                logging.info(f'resume from {ckpt_path}')
                self.load(ckpt_path)
                return True
            else:
                steps = map(lambda x: int(x.split(".")[0]), ckpts)
                step = max(steps)
        ckpt_path = os.path.join(ckpt_root, f'{step}.ckpt')
        logging.info(f'resume from {ckpt_path}')
        self.load(ckpt_path)
        return True

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

def initialize_train_state_unet(unet, config, device):
    params = []

    params += unet.parameters()
    unet_ema = unet
    unet_ema.eval()
    logging.info(f'unet has {cnt_params(unet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    train_state = TrainState(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0,
                             nnet=unet, nnet_ema=unet_ema)
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
  """Convert an integer x in (b,c,h,w) into bits in (b,n*c,h,w)."""
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

def bits2int(x, out_dtype=torch.int, n=8, c=1):
  """Converts bits x in (b,n,h,w) into an integer in (b,1,h,w)."""
  """Converts bits x in (b,n*c,h,w) into an integer in (b,c,h,w). Then reshape to (b,1,h*sqrt(c),w*sqrt(c))"""
  # c: num of patch channels, usually=1 or 4
  x = x.type(out_dtype)
  sc = int(math.sqrt(c))
  y = torch.zeros(x.shape[0],1, x.shape[2]*sc,x.shape[3]*sc,device=x.device)
  offset=[(0,0),(1,0),(0,1),(1,1)]
  for j in range(c):
    p = torch.zeros(x.shape[0],1, x.shape[2],x.shape[3],device=x.device)
    #print(x.shape, p.shape)
    for i in range(n):
        #print(p.shape, x[:,i*c+j,:,:].shape)
        p[:,0,:,:] += x[:,i*c+j,:,:] * (2 ** i)
    if c>1:
        y[:,:,offset[j][1]::sc,offset[j][0]::sc]= p 
    else:
        y=p
 
  #x = torch.sum(x * (2 ** torch.range(start=0,end=x.shape[1])), 1, keepdim=True)
  return y #.unsqueeze(1)

#NOTE: object ID = R * 256 * G + 256 * 256 + B. But
def get_colormap(path, force=False):
    if os.path.isfile(path) and force==False:  
        colormap = torch.load(path)
    else:        
        #torch.manual_seed(seed)#np.random.seed(1)
        colormap= torch.randint(0,255,(256,3))
        # save
        print('---SAVE COLORMAP---')
        torch.save(colormap, path)                 
    return colormap.cuda()
    

def color_map(x):
    #input B1HW, reshape to BHW
    if len(x.shape)>3:
        x=x.squeeze(1)
    #then map to BHW3
    colormap= get_colormap('colormap.pt')
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
def sample2dir(accelerator, path, n_samples, mini_batch_size, sample_fn, unpreprocess_fn=None, step=None, use_panoptic=False, use_ground_truth=False, mask_path=None,mask_channel=1):
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
                pred_mask= bits2int(pred_mask>0, torch.int, c=mask_channel) #this convert pred_mask to [N,1,H,W]
                #Evaluate masks by comparing bin counts
                cnt_diff= eval_mask_cnt(pred_mask, panoptic)
                eval_cnt_mask_diff.append(cnt_diff)

                color_masks= color_map(pred_mask)
            
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

                color_panoptic= color_map(panoptic[:,0,:,:])
                ground_mask = make_grid(color_panoptic.float(), 8)
                wandb.log({'ground_truth_mask': wandb.Image(ground_mask)}, step)
            for i,sample in enumerate(samples):
                #img = Image.fromarray(color_masks[i,:,:,:], 'RGB')
                #img.save(os.path.join(mask_path, f"{sample_idx[i]}.png"))
                if use_panoptic==True:
                    save_image(sample, os.path.join(path, f"{sample_idx[i]}.png"))
                    #save_image(color_masks, os.path.join(mask_path, f"{idx}.png"))
                else:
                    save_image(sample, os.path.join(path, f"{idx}.png"))
                idx += 1
    if use_panoptic==True and (step is not None) and use_ground_truth==False:
        wandb.log({f'eval_loss_mask': torch.mean(torch.stack(loss_mask_all))}, step)
        wandb.log({f'eval_cnt_mask_diff': torch.mean(torch.stack(eval_cnt_mask_diff))}, step)


def grad_norm(model):
    total_norm = 0.
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm
