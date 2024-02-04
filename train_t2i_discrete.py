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

panoptic_coco_categories = '../panopticapi-master/panoptic_coco_categories.json'

with open(panoptic_coco_categories, 'r') as f:
    categories_list = json.load(f)
categegories = {category['id']: category for category in categories_list}
color_generator=IdGenerator(categegories)

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

    def sample(self, x0, panoptic=None):  # sample from q(xn|x0), where n is uniform
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
            eps_m = torch.randn_like(panoptic) #random noise
            mask_n = stp(self.cum_alphas[n] ** 0.5, panoptic) + stp(self.cum_betas[n] ** 0.5, eps_m)
            return torch.tensor(n, device=x0.device), eps, xn, eps_m, mask_n

    def __repr__(self):
        return f'Schedule({self.betas[:10]}..., {self.N})'

#TODO: Set the flag to True to input ground truth panoptic mask to the model
use_ground_truth=True
def LSimple(x0, nnet, schedule,  loss_func, panoptic=None,**kwargs):
    if panoptic is None:
        n, eps, xn = schedule.sample(x0)  # n in {1, ..., 1000}
        eps_pred, mask_pred = nnet(xn, n, **kwargs) 
    else:
        #scale panoptic to [-1,1]
        scaled_panoptic = (panoptic/ 100.0 - 1.0) #category id's range is 1-200
        #TODO: use another noise for panoptic segmentation mask
        n, eps, xn, eps_m, mask_n = schedule.sample(x0, scaled_panoptic)  # n in {1, ..., 1000}
        #Run the diffusion model to predict noises from image xn and panoptic segmentation mask mask_n 
        #TODO: Jan17: test use ground truth panoptic mask
        if use_ground_truth==True:
            eps_pred, mask_pred = nnet(xn, n, **kwargs, mask_token=scaled_panoptic) 
        else:
            eps_pred, mask_pred = nnet(xn, n, **kwargs, mask_token=mask_n) 
    #TODO: sum the losses
    loss_eps = mos(eps - eps_pred)
    if panoptic is None:
        return loss_eps, None, None
    else:
        #Note: take the max logit as the category label
        #mask_label = torch.max(mask_pred, dim=1,keepdim=True)[1]
        panoptic=panoptic.squeeze(1).type(torch.long)
        #print('check masks',mask_pred.shape, panoptic.min(),panoptic.max()) #0-200
        if use_ground_truth==True:
            loss_mask= loss_eps
        else:
            loss_mask =  loss_func(mask_pred, panoptic).mean()
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
    test_dataset = dataset.get_split(split='test', labeled=True)  # for sampling
    test_dataset_loader = DataLoader(test_dataset, batch_size=config.sample.mini_batch_size, shuffle=True, drop_last=True,
                                     num_workers=8, pin_memory=True, persistent_workers=True)

    train_state = utils.initialize_train_state(config, device)
    nnet, nnet_ema, optimizer, train_dataset_loader, test_dataset_loader = accelerator.prepare(
        train_state.nnet, train_state.nnet_ema, train_state.optimizer, train_dataset_loader, test_dataset_loader)
    lr_scheduler = train_state.lr_scheduler
    train_state.resume(config.ckpt_root)
    loss_cross_entropy=torch.nn.CrossEntropyLoss()

    autoencoder = libs.autoencoder.get_model(**config.autoencoder)
    autoencoder.to(device)

    @ torch.cuda.amp.autocast()
    def encode(_batch):
        return autoencoder.encode(_batch)

    @ torch.cuda.amp.autocast()
    def decode(_batch):
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

    def cfg_nnet(x, timesteps, context, mask_token=None):
        _cond, pred_mask = nnet_ema(x, timesteps, context=context, mask_token=mask_token)
        _empty_context = torch.tensor(dataset.empty_context, device=device)
        _empty_context = einops.repeat(_empty_context, 'L D -> B L D', B=x.size(0))
        _uncond = nnet_ema(x, timesteps, context=_empty_context)
        return _cond + config.sample.scale * (_cond - _uncond), pred_mask

    def train_step(_batch):
        _metrics = dict()
        optimizer.zero_grad()
        _z = autoencoder.sample(_batch[0]) if 'feature' in config.dataset.name else encode(_batch[0])
        #Note: using a random initial mask, not scheduled
        #mask_token = torch.randn_like(_z,device=_z.device)
        #zero initialization
        #mask_token = torch.zeros_like(_z,device=_z.device)
        if use_panoptic==True:
            loss_eps, loss_mask= LSimple(_z, nnet, _schedule,  loss_func=loss_cross_entropy,panoptic=_batch[2],context=_batch[1])  # currently only support the extracted feature version
        else:
            loss_eps= LSimple(_z, nnet, _schedule,  panoptic=None, context=_batch[1])  # currently only support the extracted feature version
      
        _metrics['loss'] = accelerator.gather(loss_eps.detach()).mean()
        _metrics['loss_mask'] = accelerator.gather(loss_mask.detach()).mean()
        #TODO: backpropagate the sum of losses
        #TODO: Jan17 test use ground truth mask
        if use_ground_truth==True:
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
        #TODO: Jan17 use ground truth mask
        if use_ground_truth==True:
            scaled_panoptic= (panoptic/ 100.0 - 1.0)
            mask_token =  scaled_panoptic
        else:
            #initial as random
            mask_token = torch.randn(*panoptic.shape, device=device)
        #print('panoptic shape ',panoptic.shape )
        #if panoptic is not None:
        #    _z_init = _z_init* panoptic
        
        noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.tensor(_betas, device=device).float())

        def model_fn(x, t_continuous, panoptic=None, mask_token=None):
            #Note: panoptic is the ground-truth mask. It is not used in DPM solver!
            t = t_continuous * _schedule.N
            if mask_token is None:
                return cfg_nnet(x, t, **kwargs)
            else:
                return cfg_nnet(x, t, mask_token=mask_token, **kwargs)
                #TODO: try division in the backward process
                #return panoptic * cfg_nnet(x, t, **kwargs)
                #return torch.div( cfg_nnet(x, t, **kwargs), panoptic+1.0e-6)

        dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)
        solver_order=1
        #TODO: try first order solver, set order=1
        if solver_order==1:
            _z, pred_mask = dpm_solver.sample(_z_init, steps=_sample_steps, eps=1. / _schedule.N, T=1., order=1, panoptic=panoptic, method='singlestep', mask_token=mask_token)
        else:
            _z = dpm_solver.sample(_z_init, steps=_sample_steps, eps=1. / _schedule.N, T=1., order=3, panoptic=panoptic, mask_token=mask_token)
        #TODO: show predicted mask, evaluate the loss with panoptic
        if panoptic is not None:
            
            if use_ground_truth==True:
                loss_mean=1.0
            else:
                loss_mask = loss_func(pred_mask, panoptic.squeeze(1).type(torch.long)).mean()
                ##loss_mask =  mos(panoptic- pred_mask)
                loss_mean = accelerator.gather(loss_mask.detach()).mean()
            #pred_mask = make_grid(pred_mask, 8)
            #save_image(samples, os.path.join(config.sample_dir, f'{train_state.step}.png'))
            #wandb.log({'pred_mask': wandb.Image(pred_mask)}, step=train_state.step)

            return decode(_z), pred_mask, loss_mean
        else:
            return decode(_z)

    def eval_step(n_samples, sample_steps):
        logging.info(f'eval_step: n_samples={n_samples}, sample_steps={sample_steps}, algorithm=dpm_solver, '
                     f'mini_batch_size={config.sample.mini_batch_size}')

        def sample_fn(_n_samples, use_panoptic=False):
            _context = next(context_generator)
            assert _context[0].size(0) == _n_samples
            if use_panoptic==True:
                return dpm_solver_sample(_n_samples, sample_steps, panoptic=_context[1], loss_func=loss_cross_entropy, context=_context[0]) #context: conditions
            else:
                return dpm_solver_sample(_n_samples, sample_steps, panoptic=None,context=_context[0]) #context: conditions
            
        with tempfile.TemporaryDirectory() as temp_path:
            path = config.sample.path or temp_path
            if accelerator.is_main_process:
                os.makedirs(path, exist_ok=True)
            utils.sample2dir(accelerator, path, n_samples, config.sample.mini_batch_size, sample_fn, dataset.unpreprocess, step=train_state.step, use_panoptic=True)

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
    while train_state.step < config.train.n_steps:
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
            contexts = torch.tensor(dataset.contexts, device=device)[: 2 * 5]
            if use_panoptic==True:
                
                panoptic_rand = torch.zeros(10,1,32,32, dtype=torch.long, device=device)
                samples, pred_mask, _ = dpm_solver_sample(_n_samples=2*5, _sample_steps=50, panoptic=panoptic_rand, loss_func=loss_cross_entropy, context=contexts)
                mask_max, mask_label = torch.max(pred_mask,dim=1, keepdim=True) #indices=class labels, [b,1,h,w]
                #color_mask = torch.zeros_like(pred_mask)
                #logging.info('Change category ids of masks to colors')
                #for i in range(mask_label.shape[0]):
                #    color_mask[i,...] = utils.category2rgb(color_generator, mask_label[i,...],categegories)
                #TODO: print colored maps
                color_masks = torch.zeros_like(pred_mask, dtype=bool) #[b,200,h,w]
                color_masks[pred_mask==mask_max] = 1
                empty_image = torch.zeros(pred_mask.shape[0],3,32,32, dtype=torch.uint8)
                for i in range(10):
                    logging.info(f'Color masks...{i}')
                    empty_image[i,:,:,:]= draw_segmentation_masks(empty_image[i,:,:,:], color_masks[i,:,:,:], alpha=0.7)   
                grid_mask = make_grid(empty_image.float() , 5, normalize=True) 
                logging.info('Print a grid of masks...')
                #grid_mask = make_grid(mask_label.float() , 5, normalize=True)
                wandb.log({'samples pred_mask': wandb.Image(grid_mask)}, step=train_state.step)
            else:
                samples= dpm_solver_sample(_n_samples=2*5, _sample_steps=50, panoptic=None, context=contexts)
            accelerator.wait_for_everyone()
            samples = make_grid(dataset.unpreprocess(samples), 5)
            save_image(samples, os.path.join(config.sample_dir, f'{train_state.step}.png'))
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

    logging.info(f'Finish fitting, step={train_state.step}')
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
        hparams = 'coco2017-1-ground-truth-separate-attn-merge'#+date
        #hparams = 'coco2017-1-ground-truth-merge'
    return hparams


def main(argv):
    config = FLAGS.config
    config.config_name = get_config_name()
    config.hparams = get_hparams()
    config.workdir = FLAGS.workdir or os.path.join('/home/nano01/a/long273/results', config.config_name, config.hparams)
    config.ckpt_root = os.path.join(config.workdir, 'ckpts')
    config.sample_dir = os.path.join(config.workdir, 'samples')
    config.evaluation_only = FLAGS.evaluation_only
    if config.evaluation_only:
        print("!!!!evaluation only!!!!!")
    train(config)


if __name__ == "__main__":
    app.run(main)
