import ml_collections
import torch
from torch import multiprocessing as mp
from datasets import get_dataset
from torchvision.utils import make_grid, save_image, draw_segmentation_masks
import utils
import einops
from torch.utils._pytree import tree_map
import accelerate
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from dpm_solver_pp import NoiseScheduleVP, DPM_Solver
import tempfile
from tools.fid_score import calculate_fid_given_paths
from absl import logging
import builtins
import os
import wandb
import libs.autoencoder
import numpy as np
from panopticapi.utils import IdGenerator
import json
import random

#TODO: use pretrained UNET from stable diffusion 
from diffusers import UNet2DConditionModel
from diffusers import AutoencoderKL
from libs.autoencoder import DiagonalGaussianDistribution
from utils import create_unet_diffusers_config
from utils import convert_ldm_unet_checkpoint
from omegaconf import OmegaConf
from diffusers import PNDMScheduler
global use_unet
use_unet=False

def stable_diffusion_beta_schedule(linear_start=0.00085, linear_end=0.0120, n_timestep=1000):
    _betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return _betas.numpy()


def get_skip(alphas, betas):
    N = len(betas) - 1
    skip_alphas = np.ones([N + 1, N + 1], dtype=betas.dtype)
    for s in range(N + 1):
        skip_alphas[s, s + 1:] = alphas[s + 1:].cumprod()
    skip_betas = np.zeros([N + 1, N + 1], dtype=betas.dtype)
    for t in range(N + 1):
        prod = betas[1: t + 1] * skip_alphas[1: t + 1, t]
        skip_betas[:t, t] = (prod[::-1].cumsum())[::-1]
    return skip_alphas, skip_betas


def stp(s, ts: torch.Tensor):  # scalar tensor product
    if isinstance(s, np.ndarray):
        s = torch.from_numpy(s).type_as(ts)
    extra_dims = (1,) * (ts.dim() - 1)
    return s.view(-1, *extra_dims) * ts


def mos(a, start_dim=1):  # mean of square
    return a.pow(2).flatten(start_dim=start_dim).mean(dim=-1)

#TODO: global flag to use panoptic info
use_panoptic = True
p_uncond = 0. #for mask cfg
cfg_scale = 1.0

panoptic_coco_categories = '../panopticapi-master/panoptic_coco_categories.json'

with open(panoptic_coco_categories, 'r') as f:
    categories_list = json.load(f)
categegories = {category['id']: category for category in categories_list}
color_generator=IdGenerator(categegories)

#TODO: Set the flag to True to input ground truth panoptic mask to the model
use_ground_truth= False
use_twophases=False
class Schedule(object):  # discrete time
    def __init__(self, _betas):
        r""" _betas[0...999] = betas[1...1000]
             for n>=1, betas[n] is the variance of q(xn|xn-1)
             for n=0,  betas[0]=0
        """

        self._betas = _betas
        self.betas = np.append(0., _betas)
        self.alphas = 1. - self.betas
        self.N = len(_betas)

        assert isinstance(self.betas, np.ndarray) and self.betas[0] == 0
        assert isinstance(self.alphas, np.ndarray) and self.alphas[0] == 1
        assert len(self.betas) == len(self.alphas)

        # skip_alphas[s, t] = alphas[s + 1: t + 1].prod()
        self.skip_alphas, self.skip_betas = get_skip(self.alphas, self.betas)
        self.cum_alphas = self.skip_alphas[0]  # cum_alphas = alphas.cumprod()
        self.cum_betas = self.skip_betas[0]
        self.snr = self.cum_alphas / self.cum_betas
        #TODO: use panoptic info
        self.use_category_id= False

    def tilde_beta(self, s, t):
        return self.skip_betas[s, t] * self.cum_betas[s] / self.cum_betas[t]

    def sample(self, x0, panoptic=None, phaseone=True):  # sample from q(xn|x0), where n is uniform
        #TODO: for phase one, generate n in (2/N,N], for phase two, generate rand n in [1,2/N]
        '''
        if use_twophases==True:
            if phaseone==True:
                n = np.random.choice(list(range(self.N//2 + 1, self.N + 1)), (len(x0),)) 
            else: #phase two
                n = np.random.choice(list(range(1, self.N//2 + 1)), (len(x0),)) 
        else:
        '''
        n = np.random.choice(list(range(1, self.N + 1)), (len(x0),)) #random step
        eps = torch.randn_like(x0) #random noise
        # set to accumulated masked noise
        #if self.use_category_id==True and use_panoptic==True:       
        #    eps = eps #panoptic * eps
        #elif use_panoptic==True:
        #    eps = eps #panoptic * eps
        xn = stp(self.cum_alphas[n] ** 0.5, x0) + stp(self.cum_betas[n] ** 0.5, eps)
        #TODO: use another noise for panoptic segmentation mask
        if panoptic is None:
            return torch.tensor(n, device=x0.device), eps, xn
        else:
            #print('panoptic shape ',panoptic.shape ) #batch, 1, 32,32
            #if use_twophases==True:
            #    n= n*2- self.N #double the steps for mask!
            
            eps_m = torch.randn_like(panoptic) #random noise
            #TODO: inspired by SDEdit, reduce the timestep by half for masks to start from a mid point
            #n_mid = n//2
            mask_n = stp(self.cum_alphas[n] ** 0.5, panoptic) + stp(self.cum_betas[n] ** 0.5, eps_m)
            
            return torch.tensor(n, device=x0.device), eps, xn, eps_m, mask_n

    def __repr__(self):
        return f'Schedule({self.betas[:10]}..., {self.N})'


def LSimple(x0, nnet, schedule,  loss_func=mos, panoptic=None,**kwargs):
    if panoptic is None:
        n, eps, xn = schedule.sample(x0)  # n in {1, ..., 1000}
        eps_pred = nnet(xn, n, **kwargs) 
    else:
        #scale panoptic to [-1,1]
        #scaled_panoptic = (panoptic/ 100.0 - 1.0) #category id's range is 1-200
        #TODO: analog bits for masks
        # Discrete masks to analog bits.
        scaled_panoptic= utils.int2bits(panoptic,out_dtype=torch.float)
        scaled_panoptic = (scaled_panoptic * 2.0 - 1.0) # * scale =1
        #NOTE: use another noise for panoptic segmentation mask
        n, eps, xn, eps_m, mask_n = schedule.sample(x0, scaled_panoptic, phaseone=True)  # n in {1, ..., 1000}
        #TODO:CFG classifier-free guidance for mask conditions, use ground-truth mask with p=0.2
        #zero_mask = torch.zeros_like(mask_n, device=mask_n.device)
        mask_gt = random.random() < p_uncond
        #Run the diffusion model to predict noises from image xn and panoptic segmentation mask mask_n 
        if use_panoptic and mask_gt: #input ground_truth, but use regular diffusion model architecture
            eps_pred, mask_pred = nnet(xn, n, **kwargs, mask_token=scaled_panoptic, use_ground_truth=False, enable_panoptic=use_panoptic) 
        #NOTE: test use ground truth panoptic mask
        elif use_ground_truth==True:
            eps_pred, mask_pred = nnet(xn, n, **kwargs, mask_token=scaled_panoptic, use_ground_truth=True, enable_panoptic=use_panoptic) 
        else:
            eps_pred, mask_pred = nnet(xn, n, **kwargs, mask_token=mask_n, use_ground_truth=False, enable_panoptic=use_panoptic) 
            #NOTE: test alternatively optimizing eps and mask
            #eps_pred_2, mask_pred_2 = nnet(xn, n, **kwargs, mask_token=mask_pred, use_ground_truth=False, enable_panoptic=use_panoptic) 

            #TODO: test with zero mask
            #eps_pred, mask_pred = nnet(xn, n, **kwargs, mask_token=zero_mask, use_ground_truth=False, enable_panoptic=use_panoptic) 
            ##use initial noises as mask queries
            ##eps_pred, mask_pred = nnet(xn, n, **kwargs, mask_token=eps_m) 
        if use_twophases==True: #TODO:phase two use ground truth panoptic mask 
            #use same noise as phase one!
            #n2, eps2, xn2= schedule.sample(x0, phaseone=False)  # n in {1, ..., N/2}
            #TODO: use mask_pred from phase one
            mask_label = torch.max(mask_pred, dim=1,keepdim=True)[1].float()
            #scale mask input to [-1,1]. This is M0. mask_token is M[t]
            scaled_mask = mask_label/ 100.0 - 1.0 
            #eps_pred2, mask_pred2 = nnet(xn, n, **kwargs, mask_token=scaled_mask, use_ground_truth=True, enable_panoptic=True) 
            eps_pred2, mask_pred2 = nnet(xn, n, **kwargs, mask_token=scaled_panoptic, use_ground_truth=True, enable_panoptic=use_panoptic) 
            loss_2= mos(eps-eps_pred2)
    #TODO: sum the losses
    loss_eps = mos(eps - eps_pred)
    if panoptic is None:
        return loss_eps
    else:
        #Note: take the max logit as the category label
        #mask_label = torch.max(mask_pred, dim=1,keepdim=True)[1]
        panoptic=panoptic.squeeze(1).type(torch.long)
        #print('check masks',mask_pred.shape, panoptic.min(),panoptic.max()) #0-200
        if use_ground_truth==True:# or (use_panoptic and mask_gt):
            loss_mask= loss_eps
        else:
            #TODO: Use mse loss for eps mask to predict noise
            #loss_mask= mos(mask_pred-eps_m)

            #Use mse loss for analog bits
            loss_mask= mos(mask_pred-scaled_panoptic)

            #NOTE:use cross entropy loss for analog bits (8bits, target size=output size)
            #rescale from [-1,1] to [0,1]
            '''
            mask_pred= (mask_pred+1.0)/2.0 
            scaled_panoptic= (scaled_panoptic+1.0)/2.0
            loss_mask =  loss_func(mask_pred, scaled_panoptic).mean()
            '''
            #loss_mask =  loss_func(mask_pred, panoptic).mean()
        if use_twophases==True: #average losses of two phases
            return (loss_eps+loss_2)/2.0, loss_mask
        else:
            return loss_eps, loss_mask


def train(config):
    if config.get('benchmark', False):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    mp.set_start_method('spawn')
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    accelerate.utils.set_seed(config.seed, device_specific=True)
    logging.info(f'Process {accelerator.process_index} using device: {device}')

    config.mixed_precision = accelerator.mixed_precision
    config = ml_collections.FrozenConfigDict(config)

    assert config.train.batch_size % accelerator.num_processes == 0
    mini_batch_size = config.train.batch_size // accelerator.num_processes

    if accelerator.is_main_process:
        os.makedirs(config.ckpt_root, exist_ok=True)
        os.makedirs(config.sample_dir, exist_ok=True)
        os.makedirs(config.train_sample_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        wandb.init(dir=os.path.abspath(config.workdir), project=f'uvit_{config.dataset.name}', config=config.to_dict(),
                   name=config.hparams, job_type='train')#, mode='offline')
        utils.set_logger(log_level='info', fname=os.path.join(config.workdir, 'output.log'))
        logging.info(config)
    else:
        utils.set_logger(log_level='error')
        builtins.print = lambda *args: None
    logging.info(f'Run on {accelerator.num_processes} devices')

    dataset = get_dataset(**config.dataset)
    assert os.path.exists(dataset.fid_stat)
    train_dataset = dataset.get_split(split='train', labeled=True)
    train_dataset_loader = DataLoader(train_dataset, batch_size=mini_batch_size, shuffle=True, drop_last=True,
                                      num_workers=8, pin_memory=True, persistent_workers=True)
    test_dataset = dataset.get_split(split='test', labeled=True)  # for sampling, turn off shuffle
    test_dataset_loader = DataLoader(test_dataset, batch_size=config.sample.mini_batch_size, shuffle=False, drop_last=True,
                                     num_workers=8, pin_memory=True, persistent_workers=True)

    
    if config.use_unet==True: #Use pretrained unet from stable diffusion #use_safetensors=True
        #load a pretrained model from stable diffusion official
        #unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet")

        
        #NOTE: load mini-SD for 256x256 model
        original_config = OmegaConf.load('/home/min/a/long273/Documents/diffusers/U-ViT/pretrained_model/sd_finetune_256.yaml')
        checkpoint = torch.load('/home/min/a/long273/Documents/diffusers/U-ViT/pretrained_model/miniSD.ckpt')["state_dict"]
        # Convert the UNet2DConditionModel model.
        unet_config = create_unet_diffusers_config(original_config)
        converted_unet_checkpoint = convert_ldm_unet_checkpoint(checkpoint, unet_config)

        unet = UNet2DConditionModel(**unet_config)
        unet.load_state_dict(converted_unet_checkpoint,strict=False)
        del converted_unet_checkpoint
        
        if use_panoptic==True:
            unet.add_mask_stream() #add blocks for mask stream        

        train_state = utils.initialize_train_state_unet(unet, config, device)
    else:
        train_state = utils.initialize_train_state(config, device)

    nnet, nnet_ema, optimizer, train_dataset_loader, test_dataset_loader = accelerator.prepare(
        train_state.nnet, train_state.nnet_ema, train_state.optimizer, train_dataset_loader, test_dataset_loader)
    lr_scheduler = train_state.lr_scheduler
    train_state.resume(config.ckpt_root)
    
    if config.pretrained is not None: #NOTE: load a pretrained model for image generation
        nnet.load_state_dict(torch.load(config.pretrained),strict=False)
        #copy pretrained weights to mask stream
        logging.info(f'Load pretrained weights to image/mask streams')
        
        for  p1, p2 in zip (nnet.in_blocks, nnet.in_blocks_mask):
            p2.load_state_dict(p1.state_dict(),strict=False)
        nnet.mid_block_mask.load_state_dict(nnet.mid_block.state_dict(),strict=False)
        for  p1, p2 in zip (nnet.out_blocks, nnet.out_blocks_mask):
            p2.load_state_dict(p1.state_dict(),strict=False)
        
        #fix loaded weights of image stream
        
        nnet.patch_embed.requires_grad_(False)
        nnet.context_embed.requires_grad_(False)
        nnet.time_embed.requires_grad_(False)
        nnet.in_blocks.requires_grad_(False)
        nnet.mid_block.requires_grad_(False)
        nnet.out_blocks.requires_grad_(False)
    #fix pretrained weights
    '''
    if config.use_unet==True:
        #nnet.conv_in.requires_grad_(False)
        nnet.time_proj.requires_grad_(False)
        nnet.time_embedding.requires_grad_(False)
        nnet.down_blocks.requires_grad_(False)
        nnet.mid_block.requires_grad_(False)
        nnet.up_blocks.requires_grad_(False)
    '''
    loss_cross_entropy= torch.nn.CrossEntropyLoss()

    if use_unet==True:
        #runwayml/stable-diffusion-v1-5 or CompVis/stable-diffusion-v1-1
        autoencoder = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae", use_safetensors=True)        
    else:
        autoencoder = libs.autoencoder.get_model(**config.autoencoder)
    autoencoder.to(device)

    wandb.watch(nnet)

    @ torch.cuda.amp.autocast()
    def encode(_batch):
        return autoencoder.encode(_batch)
    
    @ torch.cuda.amp.autocast()
    def vae_sample(_batch):
        if use_unet==True:
            return DiagonalGaussianDistribution(_batch)
        else:
            return autoencoder.sample(_batch)

    @ torch.cuda.amp.autocast()
    def decode(_batch):
        if use_unet==True:
            global autoencoder_scale
            _batch = 1/autoencoder_scale * _batch
            return autoencoder.decode(_batch).sample
        else:
            return autoencoder.decode(_batch)

    def get_data_generator():
        while True:
            for data in tqdm(train_dataset_loader, disable=not accelerator.is_main_process, desc='epoch'):
                yield data

    data_generator = get_data_generator()

    def get_context_generator():
        while True:
            for data in test_dataset_loader:
                _context = data[1:]
                yield _context

    context_generator = get_context_generator()

    _betas = stable_diffusion_beta_schedule()
    _schedule = Schedule(_betas)
    logging.info(f'use {_schedule}')

    def cfg_nnet(x, timesteps, context, mask_token=None, use_ground_truth=False, enable_panoptic=False):

        if use_panoptic==True:
            if use_unet==True:
                _cond, pred_mask = nnet_ema(x, timesteps, encoder_hidden_states=context, mask_token=mask_token,use_ground_truth=use_ground_truth, enable_panoptic=enable_panoptic, return_dict=False)
            else:
                _cond, pred_mask = nnet_ema(x, timesteps, context=context, mask_token=mask_token,use_ground_truth=use_ground_truth, enable_panoptic=enable_panoptic)
        else:
            if use_unet==True:
                _cond= nnet_ema(x, timesteps, encoder_hidden_states=context, mask_token=None,use_ground_truth=use_ground_truth, enable_panoptic=False, return_dict=False)
            else:
                _cond= nnet_ema(x, timesteps, context=context, mask_token=None,use_ground_truth=use_ground_truth, enable_panoptic=False)
        
        _empty_context = torch.tensor(dataset.empty_context, device=device)
        _empty_context = einops.repeat(_empty_context, 'L D -> B L D', B=x.size(0))
        
        if config.sample.cfg==True:
            '''
            if enable_panoptic==True:
                if use_unet==True:
                    _uncond, pred_mask_u = nnet_ema(x, timesteps, encoder_hidden_states=_empty_context, mask_token=mask_token,use_ground_truth=use_ground_truth, enable_panoptic=enable_panoptic, return_dict=False)
                else:
                    _uncond, pred_mask_u = nnet_ema(x, timesteps, context=_empty_context, mask_token=mask_token,use_ground_truth=use_ground_truth, enable_panoptic=enable_panoptic)
            else:
            '''
            if use_unet==True: #unconditioned, no context or masks
                _uncond = nnet_ema(x, timesteps, encoder_hidden_states=_empty_context, return_dict=False)
            else:
                _uncond = nnet_ema(x, timesteps, context=_empty_context)
        
            #TODO: CFG for masks. Only context no masks
            '''
            if enable_panoptic==True:
                zero_mask = torch.zeros_like(mask_token, device=mask_token.device) if mask_token is not None else None
                if use_unet==True:
                    _uncond_mask, pred_mask_f = nnet_ema(x, timesteps, encoder_hidden_states=context, mask_token=zero_mask,use_ground_truth=use_ground_truth, enable_panoptic=use_panoptic, return_dict=False)
                else:
                    _uncond_mask, pred_mask_f = nnet_ema(x, timesteps, context=context, mask_token=zero_mask,use_ground_truth=use_ground_truth, enable_panoptic=use_panoptic)
                pred_mask = pred_mask +config.sample.scale * (pred_mask - 0.5* pred_mask_u - 0.5*pred_mask_f)
            '''
        
        if use_panoptic==True:
            #pred_mask = pred_mask +config.sample.scale * (pred_mask -  pred_mask_u)
            if config.sample.cfg==True:
                return _cond + config.sample.scale * (_cond - _uncond), pred_mask
                #return _cond + config.sample.scale * (_cond - 0.5*_uncond- 0.5*_uncond_mask), pred_mask
            else:
                return _cond, pred_mask
        else:
            if config.sample.cfg==True:
                return _cond + config.sample.scale * (_cond - _uncond), None
            else:
                return _cond, None

    def train_step(_batch):
        _metrics = dict()
        optimizer.zero_grad()
        _z = vae_sample(_batch[0]) if 'feature' in config.dataset.name else encode(_batch[0])
        #Note: using a random initial mask, not scheduled
        #mask_token = torch.randn_like(_z,device=_z.device)
        #zero initialization
        #mask_token = torch.zeros_like(_z,device=_z.device)
        if use_panoptic==True:
            if use_unet==True:
                loss_eps, loss_mask= LSimple(_z, nnet, _schedule,  loss_func=loss_cross_entropy,panoptic=_batch[2],encoder_hidden_states=_batch[1], return_dict=False)  # currently only support the extracted feature version
            else:
                loss_eps, loss_mask= LSimple(_z, nnet, _schedule,  loss_func=loss_cross_entropy,panoptic=_batch[2],context=_batch[1])  # currently only support the extracted feature version
            
            #if config.use_twophases==True:
            #    loss_eps, loss_mask, loss_2= LSimple(_z, nnet, _schedule,  loss_func=loss_cross_entropy,panoptic=_batch[2],context=_batch[1])  
        else:
            if use_unet==True:
                loss_eps= LSimple(_z, nnet, _schedule,  panoptic=None, encoder_hidden_states=_batch[1], return_dict=False)  # currently only support the extracted feature version
            else:
                loss_eps= LSimple(_z, nnet, _schedule,  panoptic=None, context=_batch[1])  # currently only support the extracted feature version
            
        _metrics['loss'] = accelerator.gather(loss_eps.detach()).mean()
        if use_panoptic==True:
            _metrics['loss_mask'] = accelerator.gather(loss_mask.detach()).mean()
        #TODO: backpropagate the sum of losses
        #TODO: Jan17 test use ground truth mask
        if use_ground_truth==True or use_panoptic==False:
            loss_sum = loss_eps 
        else:
            loss_sum = loss_eps  + loss_mask
        accelerator.backward(loss_sum.mean())
        optimizer.step()
        lr_scheduler.step()
        train_state.ema_update(config.get('ema_rate', 0.9999))
        train_state.step += 1
        return dict(lr=train_state.optimizer.param_groups[0]['lr'], **_metrics)

    def dpm_solver_sample(_n_samples, _sample_steps, panoptic=None, loss_func=None,**kwargs): #context is another input
        #TODO: modify sampling to add panoptic mask
        #add a input panoptic 
        _z_init = torch.randn(_n_samples, *config.z_shape, device=device)
        
        #TODO: use ground truth mask
        if use_panoptic==False:
            mask_token= None
        else:
            scaled_panoptic= utils.int2bits(panoptic, out_dtype=torch.float)
            scaled_panoptic = (scaled_panoptic * 2.0 - 1.0)
            if use_ground_truth==True:
                #scaled_panoptic= (panoptic/ 100.0 - 1.0)
                mask_token =  scaled_panoptic
            else:
                #initial as random
                mask_token = torch.randn(*scaled_panoptic.shape, device=device)
                #TODO: test with zero mask input 
                #mask_token = torch.zeros_like(scaled_panoptic,  device=device)
        
        #print('panoptic shape ',panoptic.shape )
        #if panoptic is not None:
        #    _z_init = _z_init* panoptic
        
        noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.tensor(_betas, device=device).float())

        def model_fn(x, t_continuous, panoptic=None, mask_token=None, use_ground_truth=False, enable_panoptic=False):
            #Note: panoptic is the ground-truth mask. It is not used in DPM solver!
            t = t_continuous * _schedule.N
            if mask_token is None:
                return cfg_nnet(x, t, **kwargs)
            else: #the arguments are enabled in dpm_solver_pp.py
                return cfg_nnet(x, t, mask_token=mask_token, use_ground_truth=use_ground_truth, enable_panoptic=enable_panoptic, **kwargs)
                #TODO: try division in the backward process
                #return panoptic * cfg_nnet(x, t, **kwargs)
                #return torch.div( cfg_nnet(x, t, **kwargs), panoptic+1.0e-6)

        dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)
        solver_order=3

        #Use Stable diffusion's scheduler!
        if use_unet==True:
            _z= _z_init

            noise_scheduler = PNDMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
            # Prepare timesteps
            noise_scheduler.set_timesteps(_sample_steps, device=device)
            timesteps = noise_scheduler.timesteps
            for i, t in enumerate(timesteps):
                if mask_token is None:
                    noise_pred, _= cfg_nnet(_z, t, **kwargs)
                else: #the arguments are enabled in dpm_solver_pp.py
                    noise_pred, pred_mask =cfg_nnet(_z, t, mask_token=mask_token, use_ground_truth=use_ground_truth, enable_panoptic=enable_panoptic, **kwargs)
                _z = noise_scheduler.step(noise_pred, t, _z).prev_sample
        else:
            #TODO: try first order solver, set order=1
            if solver_order==1:
                _z, pred_mask = dpm_solver.sample(_z_init, steps=_sample_steps, eps=1. / _schedule.N, T=1., order=1, panoptic=panoptic, method='singlestep', mask_token=mask_token, use_twophases=use_twophases, enable_mask_opt= (use_panoptic and not use_ground_truth), use_ground_truth=use_ground_truth, enable_panoptic=use_panoptic)
            else:#third order
                if use_panoptic==True:
                    _z, pred_mask = dpm_solver.sample(_z_init, steps=_sample_steps, eps=1. / _schedule.N, T=1., order=3, panoptic=panoptic, mask_token=mask_token, enable_mask_opt=(use_panoptic and not use_ground_truth), use_ground_truth=use_ground_truth, enable_panoptic=True)
                else:
                    _z, _ = dpm_solver.sample(_z_init, steps=_sample_steps, eps=1. / _schedule.N, T=1., order=3)
                
        #TODO: show predicted mask, evaluate the loss with panoptic
        if panoptic is not None:
            
            if use_ground_truth==True:
                loss_mean=1.0
            else:
                ##loss_mask = loss_func(pred_mask, panoptic.squeeze(1).type(torch.long)).mean()
                #TODO: use mos loss for analog bits
                loss_mask =  mos(scaled_panoptic- pred_mask)

                #Use cross entropy loss on analog bits; rescale from [-1,1] to [0,1]
                '''
                scaled_pred_mask= (pred_mask+1.0)/2.0
                scaled_panoptic= (scaled_panoptic+1.0)/2.0
                loss_mask = loss_func(scaled_pred_mask, scaled_panoptic).mean()
                '''
                loss_mean = accelerator.gather(loss_mask.detach()).mean()
            #pred_mask = make_grid(pred_mask, 8)
            #save_image(samples, os.path.join(config.sample_dir, f'{train_state.step}.png'))
            #wandb.log({'pred_mask': wandb.Image(pred_mask)}, step=train_state.step)

            return decode(_z), pred_mask, loss_mean, panoptic
        else:
            return decode(_z)

    def eval_step(n_samples, sample_steps):
        logging.info(f'eval_step: n_samples={n_samples}, sample_steps={sample_steps}, algorithm=dpm_solver, '
                     f'mini_batch_size={config.sample.mini_batch_size}')

        def sample_fn(_n_samples, use_panoptic=False, print_text=False):
            _context = next(context_generator)
            index= _context[2]
            if print_text==True:
                
                caption_text=""
                for i in range(index.size(0)):
                    file = open(os.path.join(config.dataset.path, f'val2017/{index[i]}_text.txt'), "r")
                    caption_text+=' '+ str(i)+ file.readline()
                with open(os.path.join(config.workdir, 'caption.log'), 'a') as f:
                    print(caption_text, file=f)
            assert _context[0].size(0) == _n_samples
            if use_panoptic==True:
                samples, pred_mask, loss_mask, panoptic= dpm_solver_sample(_n_samples, sample_steps, panoptic=_context[1], loss_func=loss_cross_entropy, context=_context[0]) #context: conditions
                return index, samples, pred_mask, loss_mask, panoptic
            else:
                return dpm_solver_sample(_n_samples, sample_steps, panoptic=None,context=_context[0]) #context: conditions
            
        with tempfile.TemporaryDirectory() as temp_path:
            #path = config.sample.path or temp_path
            #Save to the working dir of the current model
            path = config.sample_dir
            if accelerator.is_main_process:
                os.makedirs(path, exist_ok=True)
                mask_path=os.path.join(config.workdir, 'mask')

            utils.sample2dir(accelerator, path, n_samples, config.sample.mini_batch_size, sample_fn, dataset.unpreprocess, step=train_state.step, use_panoptic=use_panoptic, use_ground_truth=use_ground_truth, mask_path=mask_path, mask_channel=config.mask_channel)

            _fid = 0
            if accelerator.is_main_process:
                _fid = calculate_fid_given_paths((dataset.fid_stat, path))
                logging.info(f'step={train_state.step} fid{n_samples}={_fid}')
                with open(os.path.join(config.workdir, 'eval.log'), 'a') as f:
                    print(f'step={train_state.step} fid{n_samples}={_fid}', file=f)
                wandb.log({f'fid{n_samples}': _fid}, step=train_state.step)
                
            _fid = torch.tensor(_fid, device=device)
            _fid = accelerator.reduce(_fid, reduction='sum')

        return _fid.item()

    logging.info(f'Start fitting, step={train_state.step}, mixed_precision={config.mixed_precision}')

    step_fid = []
    while train_state.step <= config.train.n_steps or config.evaluation_only:
        #TODO: set evaluation only
        #if config.evaluation_only is None:
        nnet.train()
        batch = tree_map(lambda x: x.to(device), next(data_generator))
        metrics = train_step(batch)

        nnet.eval()
        if accelerator.is_main_process and train_state.step % config.train.log_interval == 0:
            logging.info(utils.dct2str(dict(step=train_state.step, **metrics)))
            logging.info(config.workdir)
            wandb.log(metrics, step=train_state.step)

        if accelerator.is_main_process and (train_state.step % config.train.eval_interval == 0 or config.evaluation_only):
            torch.cuda.empty_cache()
            logging.info('Save a grid of images...')
            bs= min(config.sample.mini_batch_size,10)
            contexts = torch.tensor(dataset.contexts, device=device)[: bs]
            #TODO: pick the first batch for evaluation
            for data in test_dataset_loader:
                contexts = data[1:]
            index= contexts[2][: bs]
            caption_text=""
            for i in range(index.size(0)):
                file = open(os.path.join(config.dataset.path, f'val2017/{index[i]}_text.txt'), "r")
                caption_text+=' '+ str(i)+ file.readline()
            with open(os.path.join(config.workdir, 'eval_caption.log'), 'a') as f:
                print(caption_text, file=f)
            if use_panoptic==True:
                
                panoptic_rand = torch.zeros(bs,1,32,32, dtype=torch.long, device=device)
                #TODO: check context size?
                #print('check shapes',contexts[0].shape, contexts[1].shape) #[b,7,768]
                samples, pred_mask, _, _ = dpm_solver_sample(_n_samples=bs, _sample_steps=50, panoptic=contexts[1][: bs,:,:,:], loss_func=loss_cross_entropy, context=contexts[0][: bs,:,:])
                mask_max, mask_label = torch.max(pred_mask,dim=1, keepdim=True) #indices=class labels, [b,1,h,w]
                #color_mask = torch.zeros_like(pred_mask)
                #logging.info('Change category ids of masks to colors')
                #for i in range(mask_label.shape[0]):
                #    color_mask[i,...] = utils.category2rgb(color_generator, mask_label[i,...],categegories)
                '''
                #TODO: print colored maps
                color_masks = torch.zeros_like(pred_mask, dtype=bool) #[b,200,h,w]
                color_masks[pred_mask==mask_max] = 1
                empty_image = torch.zeros(pred_mask.shape[0],3,32,32, dtype=torch.uint8)
                for i in range(10):
                    logging.info(f'Color masks...{i}')
                    empty_image[i,:,:,:]= draw_segmentation_masks(empty_image[i,:,:,:], color_masks[i,:,:,:], alpha=0.7)   
                grid_mask = make_grid(empty_image.float() , 5, normalize=True) 
                '''
                pred_mask= utils.bits2int(pred_mask>0, torch.int, c=config.mask_channel) #this convert pred_mask to [N,1,H,W]
                color_masks= utils.color_map(pred_mask)
                grid_mask = make_grid(color_masks.float() , 5) 
                logging.info('Print a grid of masks...')
                wandb.log({'samples pred_mask': wandb.Image(grid_mask)}, step=train_state.step)
            else:
                samples= dpm_solver_sample(_n_samples=bs, _sample_steps=50, panoptic=None, context=contexts[0][: bs,:,:])
            accelerator.wait_for_everyone()
            samples = make_grid(dataset.unpreprocess(samples), 5)
            save_image(samples, os.path.join(config.train_sample_dir, f'{train_state.step}.png'))
            wandb.log({'samples': wandb.Image(samples)}, step=train_state.step)
            
            torch.cuda.empty_cache()
        accelerator.wait_for_everyone()

        if config.evaluation_only or train_state.step % config.train.save_interval == 0 or train_state.step == config.train.n_steps:
            torch.cuda.empty_cache()
            accelerator.wait_for_everyone()
            fid = eval_step(n_samples=10000, sample_steps=50)  # calculate fid of the saved checkpoint
            logging.info(f'Save and eval checkpoint {train_state.step}...')
            if accelerator.local_process_index == 0:
                if len(step_fid)==0:
                    logging.info(f'Save the best checkpoint {train_state.step}...')
                    train_state.save(os.path.join(config.ckpt_root, f'{train_state.step}.ckpt'))
                else:
                    step_best = sorted(step_fid, key=lambda x: x[1])[0][0]
                    if fid<=step_best: #only save if it is the best
                        logging.info(f'Save the best checkpoint {train_state.step}...')
                        train_state.save(os.path.join(config.ckpt_root, f'{train_state.step}.ckpt'))
            accelerator.wait_for_everyone()
            
            step_fid.append((train_state.step, fid))
            torch.cuda.empty_cache()
        accelerator.wait_for_everyone()
        if config.evaluation_only:
            break

    logging.info(f'Finish fitting, step={train_state.step}')
    if config.evaluation_only:
        step_best=train_state.step
    else:
        logging.info(f'step_fid: {step_fid}')
        step_best = sorted(step_fid, key=lambda x: x[1])[0][0]
        logging.info(f'step_best: {step_best}')
    train_state.load(os.path.join(config.ckpt_root, f'{step_best}.ckpt'))
    del metrics
    accelerator.wait_for_everyone()
    eval_step(n_samples=config.sample.n_samples, sample_steps=config.sample.sample_steps)



from absl import flags
from absl import app
from ml_collections import config_flags
import sys
from pathlib import Path


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])
flags.DEFINE_string("workdir", None, "Work unit directory.")
flags.DEFINE_bool("evaluation_only", False, "skip training")


def get_config_name():
    argv = sys.argv
    for i in range(1, len(argv)):
        if argv[i].startswith('--config='):
            return Path(argv[i].split('=')[-1]).stem


def get_hparams():
    argv = sys.argv
    lst = []
    for i in range(1, len(argv)):
        #assert '=' in argv[i]
        if argv[i].startswith('--config.') and not argv[i].startswith('--config.dataset.path'):
            hparam, val = argv[i].split('=')
            hparam = hparam.split('.')[-1]
            if hparam.endswith('path'):
                val = Path(val).stem
            lst.append(f'{hparam}={val}')
    hparams = '-'.join(lst)
    if hparams == '':
        from datetime import datetime
        date= datetime.now().strftime('%Y-%m-%d-%H-%M')
        #hparams = 'coco2017-1-analog-bits-mse'#+date
        #hparams ='coco2017-1-baseline'
        #hparams = 'coco2017-3-ground-truth-analogbit'
        hparams = 'coco2017-3-mask4-finallayer'
        #hparams = 'coco2017-3-mask4-gt0.2'
        #hparams = 'coco2017-1-base-unet-pndm'
    return hparams


def main(argv):
    config = FLAGS.config
    config.config_name = get_config_name()
    config.hparams = get_hparams()
    config.workdir = FLAGS.workdir or os.path.join('/home/nano01/a/long273/results', config.config_name, config.hparams)
    config.ckpt_root = os.path.join(config.workdir, 'ckpts')
    config.sample_dir = os.path.join(config.workdir, 'samples')
    config.train_sample_dir = os.path.join(config.workdir, 'train_samples')
    config.evaluation_only = FLAGS.evaluation_only
    #config.pretrained = '/home/nano01/a/long273/results/mscoco_uvit_small/coco2017-3-cfg-mask/ckpts/300000.ckpt/nnet.pth'
    config.pretrained = '/home/min/a/long273/Documents/diffusers/U-ViT/pretrained_model/mscoco_uvit_small.pth'
    global use_unet
    use_unet=config.use_unet
    global autoencoder_scale
    autoencoder_scale=config.autoencoder.scale_factor
    if config.evaluation_only:
        print("!!!!evaluation only!!!!!")
    train(config)


if __name__ == "__main__":
    app.run(main)
