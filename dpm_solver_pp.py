import torch
import torch.nn.functional as F
import math
import numpy as np
import torch.distributed as dist
import utils


def interpolate_fn(x: torch.Tensor, xp: torch.Tensor, yp: torch.Tensor) -> torch.Tensor:
    """Performs piecewise linear interpolation for x, using xp and yp keypoints (knots).
    Performs separate interpolation for each channel.
    Args:
        x: [N, C] points to be calibrated (interpolated). Batch with C channels.
        xp: [C, K] x coordinates of the PWL knots. C is the number of channels, K is the number of knots.
        yp: [C, K] y coordinates of the PWL knots. C is the number of channels, K is the number of knots.
    Returns:
        Interpolated points of the shape [N, C].
    The piecewise linear function extends for the whole x axis (the outermost keypoints define the outermost
    infinite lines).
    For example:
    >>> calibrate1d(torch.tensor([[0.5]]), torch.tensor([[0.0, 1.0]]), torch.tensor([[0.0, 2.0]]))
    tensor([[1.0000]])
    >>> calibrate1d(torch.tensor([[-10]]), torch.tensor([[0.0, 1.0]]), torch.tensor([[0.0, 2.0]]))
    tensor([[-20.0000]])
    """
    x_breakpoints = torch.cat([x.unsqueeze(2), xp.unsqueeze(0).repeat((x.shape[0], 1, 1))], dim=2)
    num_x_points = xp.shape[1]
    sorted_x_breakpoints, x_indices = torch.sort(x_breakpoints, dim=2)
    x_idx = torch.argmin(x_indices, dim=2)
    cand_start_idx = x_idx - 1
    start_idx = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(1, device=x.device),
        torch.where(
            torch.eq(x_idx, num_x_points), torch.tensor(num_x_points - 2, device=x.device), cand_start_idx,
        ),
    )
    end_idx = torch.where(torch.eq(start_idx, cand_start_idx), start_idx + 2, start_idx + 1)
    start_x = torch.gather(sorted_x_breakpoints, dim=2, index=start_idx.unsqueeze(2)).squeeze(2).to(x.device)
    end_x = torch.gather(sorted_x_breakpoints, dim=2, index=end_idx.unsqueeze(2)).squeeze(2).to(x.device)
    start_idx2 = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(0, device=x.device),
        torch.where(
            torch.eq(x_idx, num_x_points), torch.tensor(num_x_points - 2, device=x.device), cand_start_idx,
        ),
    )
    y_positions_expanded = yp.unsqueeze(0).expand(x.shape[0], -1, -1)
    start_y = torch.gather(y_positions_expanded, dim=2, index=start_idx2.unsqueeze(2)).squeeze(2).to(x.device)
    end_y = torch.gather(y_positions_expanded, dim=2, index=(start_idx2 + 1).unsqueeze(2)).squeeze(2).to(x.device)
    cand = start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x)
    return cand


class NoiseScheduleVP:
    def __init__(self, schedule='discrete', beta_0=1e-4, beta_1=2e-2, total_N=1000, betas=None, alphas_cumprod=None):
        """Create a wrapper class for the forward SDE (VP type).

        The forward SDE ensures that the condition distribution q_{t|0}(x_t | x_0) = N ( alpha_t * x_0, sigma_t^2 * I ).
        We further define lambda_t = log(alpha_t) - log(sigma_t), which is the half-logSNR (described in the DPM-Solver paper).
        Therefore, we implement the functions for computing alpha_t, sigma_t and lambda_t. For t in [0, T], we have:

            log_alpha_t = self.marginal_log_mean_coeff(t)
            sigma_t = self.marginal_std(t)
            lambda_t = self.marginal_lambda(t)

        Moreover, as lambda(t) is an invertible function, we also support its inverse function:

            t = self.inverse_lambda(lambda_t)

        ===============================================================

        We support two types of VPSDEs: linear (DDPM) and cosine (improved-DDPM). The hyperparameters for the noise
        schedule are the default settings in DDPM and improved-DDPM:

            beta_min: A `float` number. The smallest beta for the linear schedule.
            beta_max: A `float` number. The largest beta for the linear schedule.
            cosine_s: A `float` number. The hyperparameter in the cosine schedule.
            cosine_beta_max: A `float` number. The hyperparameter in the cosine schedule.
            T: A `float` number. The ending time of the forward process.

        Note that the original DDPM (linear schedule) used the discrete-time label (0 to 999). We convert the discrete-time
        label to the continuous-time time (followed Song et al., 2021), so the beta here is 1000x larger than those in DDPM.

        ===============================================================

        Args:
            schedule: A `str`. The noise schedule of the forward SDE ('linear' or 'cosine').

        Returns:
            A wrapper object of the forward SDE (VP type).
        """
        if schedule not in ['linear', 'discrete', 'cosine']:
            raise ValueError("Unsupported noise schedule {}. The schedule needs to be 'linear' or 'cosine'".format(schedule))
        self.total_N = total_N
        self.beta_0 = beta_0 * 1000.
        self.beta_1 = beta_1 * 1000.

        if schedule == 'discrete':
            if betas is not None:
                log_alphas = 0.5 * torch.log(1 - betas).cumsum(dim=0)
            else:
                assert alphas_cumprod is not None
                log_alphas = 0.5 * torch.log(alphas_cumprod)
            self.total_N = len(log_alphas)
            self.t_discrete = torch.linspace(1. / self.total_N, 1., self.total_N).reshape((1, -1))
            self.log_alpha_discrete = log_alphas.reshape((1, -1))

        self.cosine_s = 0.008
        self.cosine_beta_max = 999.
        self.cosine_t_max = math.atan(self.cosine_beta_max * (1. + self.cosine_s) / math.pi) * 2. * (1. + self.cosine_s) / math.pi - self.cosine_s
        self.cosine_log_alpha_0 = math.log(math.cos(self.cosine_s / (1. + self.cosine_s) * math.pi / 2.))
        self.schedule = schedule
        if schedule == 'cosine':
            # For the cosine schedule, T = 1 will have numerical issues. So we manually set the ending time T.
            # Note that T = 0.9946 may be not the optimal setting. However, we find it works well.
            self.T = 0.9946
        else:
            self.T = 1.

    def marginal_log_mean_coeff(self, t):
        """
        Compute log(alpha_t) of a given continuous-time label t in [0, T].
        """
        if self.schedule == 'linear':
            return -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        elif self.schedule == 'discrete':
            return interpolate_fn(t.reshape((-1, 1)), self.t_discrete.clone().to(t.device), self.log_alpha_discrete.clone().to(t.device)).reshape((-1,))
        elif self.schedule == 'cosine':
            log_alpha_fn = lambda s: torch.log(torch.cos((s + self.cosine_s) / (1. + self.cosine_s) * math.pi / 2.))
            log_alpha_t =  log_alpha_fn(t) - self.cosine_log_alpha_0
            return log_alpha_t
        else:
            raise ValueError("Unsupported ")

    def marginal_alpha(self, t):
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        """
        Compute sigma_t of a given continuous-time label t in [0, T].
        """
        return torch.sqrt(1. - torch.exp(2. * self.marginal_log_mean_coeff(t)))

    def marginal_lambda(self, t):
        """
        Compute lambda_t = log(alpha_t) - log(sigma_t) of a given continuous-time label t in [0, T].
        """
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_std = 0.5 * torch.log(1. - torch.exp(2. * log_mean_coeff))
        return log_mean_coeff - log_std

    def inverse_lambda(self, lamb):
        """
        Compute the continuous-time label t in [0, T] of a given half-logSNR lambda_t.
        """
        if self.schedule == 'linear':
            tmp = 2. * (self.beta_1 - self.beta_0) * torch.logaddexp(-2. * lamb, torch.zeros((1,)).to(lamb))
            Delta = self.beta_0**2 + tmp
            return tmp / (torch.sqrt(Delta) + self.beta_0) / (self.beta_1 - self.beta_0)
        elif self.schedule == 'discrete':
            log_alpha = -0.5 * torch.logaddexp(torch.zeros((1,)).to(lamb.device), -2. * lamb)
            t = interpolate_fn(log_alpha.reshape((-1, 1)), torch.flip(self.log_alpha_discrete.clone().to(lamb.device), [1]), torch.flip(self.t_discrete.clone().to(lamb.device), [1]))
            return t.reshape((-1,))
        else:
            log_alpha = -0.5 * torch.logaddexp(-2. * lamb, torch.zeros((1,)).to(lamb))
            t_fn = lambda log_alpha_t: torch.arccos(torch.exp(log_alpha_t + self.cosine_log_alpha_0)) * 2. * (1. + self.cosine_s) / math.pi - self.cosine_s
            t = t_fn(log_alpha)
            return t


def model_wrapper(model, noise_schedule=None, is_cond_classifier=False, classifier_fn=None, classifier_scale=1., time_input_type='1', total_N=1000, model_kwargs={}, is_deis=False):
    """Create a wrapper function for the noise prediction model.

    DPM-Solver needs to solve the continuous-time diffusion ODEs. For DPMs trained on discrete-time labels, we need to
    firstly wrap the model function to a function that accepts the continuous time as the input.

    The input `model` has the following format:

    ``
        model(x, t_input, **model_kwargs) -> noise
    ``

    where `x` and `noise` have the same shape, and `t_input` is the time label of the model.
    (may be discrete-time labels (i.e. 0 to 999) or continuous-time labels (i.e. epsilon to T).)

    We wrap the model function to the following format:

    ``
        def model_fn(x, t_continuous) -> noise:
            t_input = get_model_input_time(t_continuous)
            return model(x, t_input, **model_kwargs)            
    ``
    
    where `t_continuous` is the continuous time labels (i.e. epsilon to T). And we use `model_fn` for DPM-Solver.

    For DPMs with classifier guidance, we also combine the model output with the classifier gradient as used in [1].

    [1] P. Dhariwal and A. Q. Nichol, "Diffusion models beat GANs on image synthesis," in Advances in Neural 
    Information Processing Systems, vol. 34, 2021, pp. 8780-8794.

    ===============================================================

    Args:
        model: A noise prediction model with the following format:
            ``
                def model(x, t_input, **model_kwargs):
                    return noise
            ``
        noise_schedule: A noise schedule object, such as NoiseScheduleVP. Only used for the classifier guidance.
        is_cond_classifier: A `bool`. Whether to use the classifier guidance.
        classifier_fn: A classifier function. Only used for the classifier guidance. The format is:
            ``
                def classifier_fn(x, t_input):
                    return logits
            ``
        classifier_scale: A `float`. The scale for the classifier guidance.
        time_input_type: A `str`. The type for the time input of the model. We support three types:
            - '0': The continuous-time type. In this case, the model is trained on the continuous time,
                so `t_input` = `t_continuous`.
            - '1': The Type-1 discrete type described in the Appendix of DPM-Solver paper.
                **For discrete-time DPMs, we recommend to use this type for DPM-Solver**.
            - '2': The Type-2 discrete type described in the Appendix of DPM-Solver paper.
        total_N: A `int`. The total number of the discrete-time DPMs (default is 1000), used when `time_input_type`
            is '1' or '2'.
        model_kwargs: A `dict`. A dict for the other inputs of the model function.
    Returns:
        A function that accepts the continuous time as the input, with the following format:
            ``
                def model_fn(x, t_continuous):
                    t_input = get_model_input_time(t_continuous)
                    return model(x, t_input, **model_kwargs)            
            ``
    """
    def get_model_input_time(t_continuous):
        """
        Convert the continuous-time `t_continuous` (in [epsilon, T]) to the model input time.
        """
        if time_input_type == '0':
            # discrete_type == '0' means that the model is continuous-time model.
            # For continuous-time DPMs, the continuous time equals to the discrete time.
            return t_continuous
        elif time_input_type == '1':
            # Type-1 discrete label, as detailed in the Appendix of DPM-Solver.
            return 1000. * torch.max(t_continuous - 1. / total_N, torch.zeros_like(t_continuous).to(t_continuous))
        elif time_input_type == '2':
            # Type-2 discrete label, as detailed in the Appendix of DPM-Solver.
            max_N = (total_N - 1) / total_N * 1000.
            return max_N * t_continuous
        else:
            raise ValueError("Unsupported time input type {}, must be '0' or '1' or '2'".format(time_input_type))

    def cond_fn(x, t_discrete, y):
        """
        Compute the gradient of the classifier, multiplied with the sclae of the classifier guidance. 
        """
        assert y is not None
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier_fn(x_in, t_discrete)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return classifier_scale * torch.autograd.grad(selected.sum(), x_in)[0]

    def model_fn(x, t_continuous):
        """
        The noise predicition model function that is used for DPM-Solver.
        """
        if t_continuous.reshape((-1,)).shape[0] == 1:
            t_continuous = torch.ones((x.shape[0],)).to(x.device) * t_continuous
        if is_cond_classifier:
            y = model_kwargs.get("y", None)
            if y is None:
                raise ValueError("For classifier guidance, the label y has to be in the input.")
            t_discrete = get_model_input_time(t_continuous)
            noise_uncond = model(x, t_discrete, **model_kwargs)
            cond_grad = cond_fn(x, t_discrete, y)
            if is_deis:
                sigma_t = noise_schedule.marginal_std(t_continuous / 1000.)
            else:
                sigma_t = noise_schedule.marginal_std(t_continuous)
            dims = len(cond_grad.shape) - 1
            return noise_uncond - sigma_t[(...,) + (None,)*dims] * cond_grad
        else:
            t_discrete = get_model_input_time(t_continuous)
            return model(x, t_discrete, **model_kwargs)

    return model_fn


class DPM_Solver:
    def __init__(self, model_fn, noise_schedule, predict_x0=False, thresholding=False, max_val=1.):
        """Construct a DPM-Solver. 

        Args:
            model_fn: A noise prediction model function which accepts the continuous-time input
                (t in [epsilon, T]):
                ``
                def model_fn(x, t_continuous):
                    return noise
                ``
            noise_schedule: A noise schedule object, such as NoiseScheduleVP.
        """
        self.model = model_fn
        self.noise_schedule = noise_schedule
        self.predict_x0 = predict_x0
        self.thresholding = thresholding
        self.max_val = max_val

    def model_fn(self, x, t, panoptic=None, mask_token=None,  use_ground_truth=False, enable_panoptic=False):
        #TODO: edit this to give panoptic segment info
        if self.predict_x0:
            alpha_t, sigma_t = self.noise_schedule.marginal_alpha(t), self.noise_schedule.marginal_std(t)
            noise, pred_mask = self.model(x, t, panoptic=panoptic, mask_token=mask_token, use_ground_truth=use_ground_truth, enable_panoptic=enable_panoptic)
            dims = len(x.shape) - 1
            x0 = (x - sigma_t[(...,) + (None,)*dims] * noise) / alpha_t[(...,) + (None,)*dims]
            #TODO: Use mse loss for eps mask to predict noise
            #mask_0=  (mask_token - sigma_t[(...,) + (None,)*dims] * pred_mask) / alpha_t[(...,) + (None,)*dims]
            if self.thresholding:
                p = 0.995
                s = torch.quantile(torch.abs(x0).reshape((x0.shape[0], -1)), p, dim=1)
                s = torch.maximum(s, torch.ones_like(s).to(s.device))[(...,) + (None,)*dims]
                x0 = torch.clamp(x0, -s, s) / (s / self.max_val)
            
            #return x0, mask_0
            return x0, pred_mask
        else:
            return self.model(x, t, panoptic=panoptic, mask_token=mask_token, use_ground_truth=use_ground_truth, enable_panoptic=enable_panoptic)

    def get_time_steps(self, skip_type, t_T, t_0, N, device):
        """Compute the intermediate time steps for sampling.

        Args:
            skip_type: A `str`. The type for the spacing of the time steps. We support three types:
                - 'logSNR': uniform logSNR for the time steps, **recommended for DPM-Solver**.
                - 'time_uniform': uniform time for the time steps. (Used in DDIM and DDPM.)
                - 'time_quadratic': quadratic time for the time steps. (Used in DDIM for low-resolutional data.)
            t_T: A `float`. The starting time of the sampling (default is T).
            t_0: A `float`. The ending time of the sampling (default is epsilon).
            N: A `int`. The total number of the spacing of the time steps.
            device: A torch device.
        Returns:
            A pytorch tensor of the time steps, with the shape (N + 1,).
        """
        if skip_type == 'logSNR':
            lambda_T = self.noise_schedule.marginal_lambda(torch.tensor(t_T).to(device))
            lambda_0 = self.noise_schedule.marginal_lambda(torch.tensor(t_0).to(device))
            logSNR_steps = torch.linspace(lambda_T.cpu().item(), lambda_0.cpu().item(), N + 1).to(device)
            # print(torch.min(torch.abs(logSNR_steps - self.noise_schedule.marginal_lambda(self.noise_schedule.inverse_lambda(logSNR_steps)))).item())
            return self.noise_schedule.inverse_lambda(logSNR_steps)
        elif skip_type == 't2':
            t_order = 2
            t = torch.linspace(t_T**(1. / t_order), t_0**(1. / t_order), N + 1).pow(t_order).to(device)
            return t
        elif skip_type == 'time_uniform':
            return torch.linspace(t_T, t_0, N + 1).to(device)
        elif skip_type == 'time_quadratic':
            t = torch.linspace(t_0, t_T, 10000000).to(device)
            quadratic_t = torch.sqrt(t)
            quadratic_steps = torch.linspace(quadratic_t[0], quadratic_t[-1], N + 1).to(device)
            return torch.flip(torch.cat([t[torch.searchsorted(quadratic_t, quadratic_steps)[:-1]], t_T * torch.ones((1,)).to(device)], dim=0), dims=[0])
        else:
            raise ValueError("Unsupported skip_type {}, need to be 'logSNR' or 'time_uniform' or 'time_quadratic'".format(skip_type))

    def get_time_steps_for_dpm_solver_fast(self, skip_type, t_T, t_0, steps, order, device):
        """
        Compute the intermediate time steps and the order of each step for sampling by DPM-Solver-fast.

        We recommend DPM-Solver-fast for fast sampling of DPMs. Given a fixed number of function evaluations by `steps`,
        the sampling procedure by DPM-Solver-fast is:
            - Denote K = (steps // 3 + 1). We take K intermediate time steps for sampling.
            - If steps % 3 == 0, we use (K - 2) steps of DPM-Solver-3, and 1 step of DPM-Solver-2 and 1 step of DPM-Solver-1.
            - If steps % 3 == 1, we use (K - 1) steps of DPM-Solver-3 and 1 step of DPM-Solver-1.
            - If steps % 3 == 2, we use (K - 1) steps of DPM-Solver-3 and 1 step of DPM-Solver-2.

        ============================================
        Args:
            t_T: A `float`. The starting time of the sampling (default is T).
            t_0: A `float`. The ending time of the sampling (default is epsilon).
            steps: A `int`. The total number of function evaluations (NFE).
            device: A torch device.
        Returns:
            orders: A list of the solver order of each step.
            timesteps: A pytorch tensor of the time steps, with the shape of (K + 1,).
        """
        if order == 3:
            K = steps // 3 + 1
            if steps % 3 == 0:
                orders = [3,] * (K - 2) + [2, 1]
            elif steps % 3 == 1:
                orders = [3,] * (K - 1) + [1]
            else:
                orders = [3,] * (K - 1) + [2]
            timesteps = self.get_time_steps(skip_type, t_T, t_0, K, device)
            return orders, timesteps
        elif order == 2:
            K = steps // 2
            if steps % 2 == 0:
                orders = [2,] * K
            else:
                orders = [2,] * K + [1]
            timesteps = self.get_time_steps(skip_type, t_T, t_0, K, device)
            return orders, timesteps
        else:
            raise ValueError("order must >= 2")

    def denoise_fn(self, x, s, noise_s=None):
        ns = self.noise_schedule
        dims = len(x.shape) - 1
        log_alpha_s = ns.marginal_log_mean_coeff(s)
        sigma_s = ns.marginal_std(s)

        if noise_s is None:
            noise_s = self.model_fn(x, s)
        x_0 = (
            (x - sigma_s[(...,) + (None,)*dims] * noise_s) / torch.exp(log_alpha_s)[(...,) + (None,)*dims]
        )
        return x_0

    def dpm_solver_first_update(self, x, s, t, noise_s=None, return_noise=False, panoptic=None, mask_token=None, enable_mask_opt=True,  use_ground_truth=False, enable_panoptic=False):
        """
        A single step for DPM-Solver-1.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (x.shape[0],).
            t: A pytorch tensor. The ending time, with the shape (x.shape[0],).
            return_noise: A `bool`. If true, also return the predicted noise at time `s`.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        ns = self.noise_schedule
        dims = len(x.shape) - 1
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        log_alpha_s, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(t)
        sigma_s, sigma_t = ns.marginal_std(s), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)

        if self.predict_x0:
            phi_1 = (torch.exp(-h) - 1.) / (-1.)
            if noise_s is None:
                noise_s, pred_mask = self.model_fn(x, s, panoptic=panoptic, mask_token=mask_token,  use_ground_truth=use_ground_truth, enable_panoptic=enable_panoptic) #noise_s is x0 when predict_x0=true
            x_t = (
                (sigma_t / sigma_s)[(...,) + (None,)*dims] * x
                + (alpha_t * phi_1)[(...,) + (None,)*dims] * noise_s
            ) #x: x at previous t; noise_s: x0  
            #compute mask t, the iterative input for the next step
            #Note: use mask labels
            if enable_mask_opt==True:
                
                #Directly use analog bits to generate the next mask
                mask_t = (
                    (sigma_t / sigma_s)[(...,) + (None,)*dims] * mask_token
                    + (alpha_t * phi_1)[(...,) + (None,)*dims] * pred_mask 
                )
                return x_t, pred_mask, mask_t #pred_mask size=[B,8,H,W], mast_t size=[B,8,H,W]
                '''
                #Get mask category ids from max idx
                mask_label = torch.max(pred_mask, dim=1,keepdim=True)[1].float()
                #scale mask input to [-1,1]. This is M0. mask_token is M[t]
                scaled_mask = mask_label/ 100.0 - 1.0 #category id's range is 1-200
                
                #Test use mask_t directly since we optimize loss(m0, pred_mask) directly
                #mask_t= scaled_mask
                
                mask_t = (
                    (sigma_t / sigma_s)[(...,) + (None,)*dims] * mask_token
                    + (alpha_t * phi_1)[(...,) + (None,)*dims] * scaled_mask #mask_label
                )
                '''
            if return_noise:
                return x_t, {'noise_s': noise_s}
            else:
                return x_t, pred_mask, pred_mask
        else:
            phi_1 = torch.expm1(h)
            if noise_s is None:
                noise_s, pred_mask = self.model_fn(x, s, panoptic=panoptic, mask_token=mask_token)
            x_t = (
                torch.exp(log_alpha_t - log_alpha_s)[(...,) + (None,)*dims] * x
                - (sigma_t * phi_1)[(...,) + (None,)*dims] * noise_s
            )
            #Note: use mask labels
            mask_label = torch.max(pred_mask, dim=1,keepdim=True)[1].float()#scale mask input to [-1,1]. This is M0. mask_token is M[t]
            scaled_mask = mask_label/ 100.0 - 1.0#category id's range is 1-200
            mask_t = (
                torch.exp(log_alpha_t - log_alpha_s)[(...,) + (None,)*dims] * mask_token
                - (sigma_t * phi_1)[(...,) + (None,)*dims] * scaled_mask
            )
            if return_noise:
                return x_t, {'noise_s': noise_s}
            else:
                return x_t, pred_mask,mask_t

    def dpm_solver_second_update(self, x, s, t, r1=0.5, noise_s=None, return_noise=False, solver_type='dpm_solver', panoptic=None, mask_token=None, enable_mask_opt=True,  use_ground_truth=False, enable_panoptic=False):
        """
        A single step for DPM-Solver-2.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (x.shape[0],).
            t: A pytorch tensor. The ending time, with the shape (x.shape[0],).
            r1: A `float`. The hyperparameter of the second-order solver. We recommend the default setting `0.5`.
            noise_s: A pytorch tensor. The predicted noise at time `s`.
                If `noise_s` is None, we compute the predicted noise by `x` and `s`; otherwise we directly use it.
            return_noise: A `bool`. If true, also return the predicted noise at time `s` and `s1` (the intermediate time).
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        if r1 is None:
            r1 = 0.5
        ns = self.noise_schedule
        dims = len(x.shape) - 1
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        lambda_s1 = lambda_s + r1 * h
        s1 = ns.inverse_lambda(lambda_s1)
        log_alpha_s, log_alpha_s1, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(s1), ns.marginal_log_mean_coeff(t)
        sigma_s, sigma_s1, sigma_t = ns.marginal_std(s), ns.marginal_std(s1), ns.marginal_std(t)
        alpha_s1, alpha_t = torch.exp(log_alpha_s1), torch.exp(log_alpha_t)

        if self.predict_x0:
            phi_11 = torch.expm1(-r1 * h)
            phi_1 = torch.expm1(-h)

            if noise_s is None:
                noise_s, pred_mask = self.model_fn(x, s, panoptic=panoptic, mask_token=mask_token,  use_ground_truth=use_ground_truth, enable_panoptic=enable_panoptic)
            x_s1 = (
                (sigma_s1 / sigma_s)[(...,) + (None,)*dims] * x
                - (alpha_s1 * phi_11)[(...,) + (None,)*dims] * noise_s
            )

            if enable_mask_opt==True:                
                #TODO: multi-step. first step for generating mask
                mask_s1 = (
                    (sigma_s1 / sigma_s)[(...,) + (None,)*dims] * mask_token
                    + (alpha_s1 * phi_11)[(...,) + (None,)*dims] * pred_mask 
                )
            else:
                mask_s1 = mask_token

            noise_s1, pred_mask_s1 = self.model_fn(x_s1, s1, panoptic=panoptic, mask_token=mask_s1,  use_ground_truth=use_ground_truth, enable_panoptic=enable_panoptic)
            if solver_type == 'dpm_solver':
                x_t = (
                    (sigma_t / sigma_s)[(...,) + (None,)*dims] * x
                    - (alpha_t * phi_1)[(...,) + (None,)*dims] * noise_s
                    - (0.5 / r1) * (alpha_t * phi_1)[(...,) + (None,)*dims] * (noise_s1 - noise_s)
                )
                if enable_mask_opt==True:                
                    #TODO: multi-step. second step for generating mask
                    mask_t = ((sigma_t / sigma_s)[(...,) + (None,)*dims] * mask_token
                        - (alpha_t * phi_1)[(...,) + (None,)*dims] * pred_mask 
                        - (0.5 / r1) * (alpha_t * phi_1)[(...,) + (None,)*dims] * (pred_mask_s1 - pred_mask )
                    )
                else:
                    mask_t = mask_token
                
            elif solver_type == 'taylor':
                x_t = (
                    (sigma_t / sigma_s)[(...,) + (None,)*dims] * x
                    - (alpha_t * phi_1)[(...,) + (None,)*dims] * noise_s
                    + (1. / r1) * (alpha_t * ((torch.exp(-h) - 1.) / h + 1.))[(...,) + (None,)*dims] * (noise_s1 - noise_s)
                )
            else:
                raise ValueError("solver_type must be either dpm_solver or taylor, got {}".format(solver_type))
        else:
            phi_11 = torch.expm1(r1 * h)
            phi_1 = torch.expm1(h)

            if noise_s is None:
                noise_s = self.model_fn(x, s, panoptic=panoptic, mask_token=mask_token)
            x_s1 = (
                torch.exp(log_alpha_s1 - log_alpha_s)[(...,) + (None,)*dims] * x
                - (sigma_s1 * phi_11)[(...,) + (None,)*dims] * noise_s
            )
            noise_s1 = self.model_fn(x_s1, s1, panoptic=panoptic, mask_token=mask_token)
            if solver_type == 'dpm_solver':
                x_t = (
                    torch.exp(log_alpha_t - log_alpha_s)[(...,) + (None,)*dims] * x
                    - (sigma_t * phi_1)[(...,) + (None,)*dims] * noise_s
                    - (0.5 / r1) * (sigma_t * phi_1)[(...,) + (None,)*dims] * (noise_s1 - noise_s)
                )
            elif solver_type == 'taylor':
                x_t = (
                    torch.exp(log_alpha_t - log_alpha_s)[(...,) + (None,)*dims] * x
                    - (sigma_t * phi_1)[(...,) + (None,)*dims] * noise_s
                    - (1. / r1) * (sigma_t * ((torch.exp(h) - 1.) / h - 1.))[(...,) + (None,)*dims] * (noise_s1 - noise_s)
                )
            else:
                raise ValueError("solver_type must be either dpm_solver or taylor, got {}".format(solver_type))
        if return_noise:
            return x_t, {'noise_s': noise_s, 'noise_s1': noise_s1}
        else:
            if enable_mask_opt==True:

                return x_t, pred_mask, mask_t
            else:
                return x_t, pred_mask, pred_mask                


    def dpm_multistep_second_update(self, x, noise_prev_list, t_prev_list, t, solver_type="dpm_solver"):
        ns = self.noise_schedule
        dims = len(x.shape) - 1
        noise_prev_1, noise_prev_0 = noise_prev_list
        t_prev_1, t_prev_0 = t_prev_list
        lambda_prev_1, lambda_prev_0, lambda_t = ns.marginal_lambda(t_prev_1), ns.marginal_lambda(t_prev_0), ns.marginal_lambda(t)
        log_alpha_prev_0, log_alpha_t = ns.marginal_log_mean_coeff(t_prev_0), ns.marginal_log_mean_coeff(t)
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)

        h_0 = lambda_prev_0 - lambda_prev_1
        h = lambda_t - lambda_prev_0
        r0 = h_0 / h
        D1_0 = (1. / r0)[(...,) + (None,)*dims] * (noise_prev_0 - noise_prev_1)
        if self.predict_x0:
            if solver_type == 'taylor':
                x_t = (
                    (sigma_t / sigma_prev_0)[(...,) + (None,)*dims] * x
                    - (alpha_t * (torch.exp(-h) - 1.))[(...,) + (None,)*dims] * noise_prev_0
                    + (alpha_t * ((torch.exp(-h) - 1.) / h + 1.))[(...,) + (None,)*dims] * D1_0
                )
            elif solver_type == 'dpm_solver':
                x_t = (
                    (sigma_t / sigma_prev_0)[(...,) + (None,)*dims] * x
                    - (alpha_t * (torch.exp(-h) - 1.))[(...,) + (None,)*dims] * noise_prev_0
                    - 0.5 * (alpha_t * (torch.exp(-h) - 1.))[(...,) + (None,)*dims] * D1_0
                )
        else:
            if solver_type == 'taylor':
                x_t = (
                    torch.exp(log_alpha_t - log_alpha_prev_0)[(...,) + (None,)*dims] * x
                    - (sigma_t * (torch.exp(h) - 1.))[(...,) + (None,)*dims] * noise_prev_0
                    - (sigma_t * ((torch.exp(h) - 1.) / h - 1.))[(...,) + (None,)*dims] * D1_0
                )
            elif solver_type == 'dpm_solver':
                x_t = (
                    torch.exp(log_alpha_t - log_alpha_prev_0)[(...,) + (None,)*dims] * x
                    - (sigma_t * (torch.exp(h) - 1.))[(...,) + (None,)*dims] * noise_prev_0
                    - 0.5 * (sigma_t * (torch.exp(h) - 1.))[(...,) + (None,)*dims] * D1_0
                )
        return x_t


    def dpm_multistep_third_update(self, x, noise_prev_list, t_prev_list, t, solver_type='dpm_solver'):
        ns = self.noise_schedule
        dims = len(x.shape) - 1
        noise_prev_2, noise_prev_1, noise_prev_0 = noise_prev_list
        t_prev_2, t_prev_1, t_prev_0 = t_prev_list
        lambda_prev_2, lambda_prev_1, lambda_prev_0, lambda_t = ns.marginal_lambda(t_prev_2), ns.marginal_lambda(t_prev_1), ns.marginal_lambda(t_prev_0), ns.marginal_lambda(t)
        log_alpha_prev_0, log_alpha_t = ns.marginal_log_mean_coeff(t_prev_0), ns.marginal_log_mean_coeff(t)
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)

        h_1 = lambda_prev_1 - lambda_prev_2
        h_0 = lambda_prev_0 - lambda_prev_1
        h = lambda_t - lambda_prev_0
        r0, r1 = h_0 / h, h_1 / h
        D1_0 = (1. / r0)[(...,) + (None,)*dims] * (noise_prev_0 - noise_prev_1)
        D1_1 = (1. / r1)[(...,) + (None,)*dims] * (noise_prev_1 - noise_prev_2)
        D1 = D1_0 + (r0 / (r0 + r1))[(...,) + (None,)*dims] * (D1_0 - D1_1)
        D2 = (1. / (r0 + r1))[(...,) + (None,)*dims] * (D1_0 - D1_1)
        if self.predict_x0:
            x_t = (
                (sigma_t / sigma_prev_0)[(...,) + (None,)*dims] * x
                - (alpha_t * (torch.exp(-h) - 1.))[(...,) + (None,)*dims] * noise_prev_0
                + (alpha_t * ((torch.exp(-h) - 1.) / h + 1.))[(...,) + (None,)*dims] * D1
                - (alpha_t * ((torch.exp(-h) - 1. + h) / h**2 - 0.5))[(...,) + (None,)*dims] * D2
            )
        else:
            x_t = (
                torch.exp(log_alpha_t - log_alpha_prev_0)[(...,) + (None,)*dims] * x
                - (sigma_t * (torch.exp(h) - 1.))[(...,) + (None,)*dims] * noise_prev_0
                - (sigma_t * ((torch.exp(h) - 1.) / h - 1.))[(...,) + (None,)*dims] * D1
                - (sigma_t * ((torch.exp(h) - 1. - h) / h**2 - 0.5))[(...,) + (None,)*dims] * D2
            )
        return x_t

    def dpm_solver_third_update(self, x, s, t, r1=1./3., r2=2./3., noise_s=None, noise_s1=None, noise_s2=None, return_noise=False, solver_type='dpm_solver', panoptic=None, mask_token=None, enable_mask_opt=True,  use_ground_truth=False, enable_panoptic=False):
        """
        A single step for DPM-Solver-3.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (x.shape[0],).
            t: A pytorch tensor. The ending time, with the shape (x.shape[0],).
            r1: A `float`. The hyperparameter of the third-order solver. We recommend the default setting `1 / 3`.
            r2: A `float`. The hyperparameter of the third-order solver. We recommend the default setting `2 / 3`.
            noise_s: A pytorch tensor. The predicted noise at time `s`.
                If `noise_s` is None, we compute the predicted noise by `x` and `s`; otherwise we directly use it.
            noise_s1: A pytorch tensor. The predicted noise at time `s1` (the intermediate time given by `r1`).
                If `noise_s1` is None, we compute the predicted noise by `s1`; otherwise we directly use it.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        if r1 is None:
            r1 = 1. / 3.
        if r2 is None:
            r2 = 2. / 3.
        ns = self.noise_schedule
        dims = len(x.shape) - 1
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        lambda_s1 = lambda_s + r1 * h
        lambda_s2 = lambda_s + r2 * h
        s1 = ns.inverse_lambda(lambda_s1)
        s2 = ns.inverse_lambda(lambda_s2)
        log_alpha_s, log_alpha_s1, log_alpha_s2, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(s1), ns.marginal_log_mean_coeff(s2), ns.marginal_log_mean_coeff(t)
        sigma_s, sigma_s1, sigma_s2, sigma_t = ns.marginal_std(s), ns.marginal_std(s1), ns.marginal_std(s2), ns.marginal_std(t)
        alpha_s1, alpha_s2, alpha_t = torch.exp(log_alpha_s1), torch.exp(log_alpha_s2), torch.exp(log_alpha_t)

        if self.predict_x0:
            phi_11 = torch.expm1(-r1 * h)
            phi_12 = torch.expm1(-r2 * h)
            phi_1 = torch.expm1(-h)
            phi_22 = torch.expm1(-r2 * h) / (r2 * h) + 1.
            phi_2 = phi_1 / h + 1.
            phi_3 = phi_2 / h - 0.5

            if noise_s is None:
                noise_s, pred_mask = self.model_fn(x, s, panoptic=panoptic, mask_token=mask_token,  use_ground_truth=use_ground_truth, enable_panoptic=enable_panoptic)
            
            if noise_s1 is None:
                x_s1 = (
                    (sigma_s1 / sigma_s)[(...,) + (None,)*dims] * x
                    - (alpha_s1 * phi_11)[(...,) + (None,)*dims] * noise_s
                )
                if enable_mask_opt==True:                
                    #TODO: multi-step. first step for generating mask
                    mask_s1 = (
                        (sigma_s1 / sigma_s)[(...,) + (None,)*dims] * mask_token
                        + (alpha_s1 * phi_11)[(...,) + (None,)*dims] * pred_mask 
                    )
                else:
                    mask_s1 = mask_token
                noise_s1, pred_mask_s1 = self.model_fn(x_s1, s1, panoptic=panoptic, mask_token=mask_s1,  use_ground_truth=use_ground_truth, enable_panoptic=enable_panoptic)
            if noise_s2 is None:
                x_s2 = (
                    (sigma_s2 / sigma_s)[(...,) + (None,)*dims] * x
                    - (alpha_s2 * phi_12)[(...,) + (None,)*dims] * noise_s
                    + r2 / r1 * (alpha_s2 * phi_22)[(...,) + (None,)*dims] * (noise_s1 - noise_s)
                )
                if enable_mask_opt==True:                
                    #TODO: multi-step. second step for generating mask
                    mask_s2 = ((sigma_s2 / sigma_s)[(...,) + (None,)*dims] * mask_token
                        - (alpha_s2 * phi_12)[(...,) + (None,)*dims] * pred_mask 
                         + r2 / r1 * (alpha_s2 * phi_22)[(...,) + (None,)*dims] * (pred_mask_s1 - pred_mask )
                    )
                else:
                    mask_s2 = mask_token

                noise_s2, pred_mask_s2 = self.model_fn(x_s2, s2, panoptic=panoptic, mask_token=mask_s2,  use_ground_truth=use_ground_truth, enable_panoptic=enable_panoptic)
            if solver_type == 'dpm_solver':
                x_t = (
                    (sigma_t / sigma_s)[(...,) + (None,)*dims] * x
                    - (alpha_t * phi_1)[(...,) + (None,)*dims] * noise_s
                    + (1. / r2) * (alpha_t * phi_2)[(...,) + (None,)*dims] * (noise_s2 - noise_s)
                )
                if enable_mask_opt==True:                
                    #TODO: multi-step. second step for generating mask
                    mask_t = ((sigma_t / sigma_s)[(...,) + (None,)*dims] * mask_token
                        - (alpha_t * phi_1)[(...,) + (None,)*dims] * pred_mask 
                        + (1. / r2) * (alpha_t * phi_2)[(...,) + (None,)*dims] * (pred_mask_s2 - pred_mask )
                    )
                else:
                    mask_t = mask_token
            elif solver_type == 'taylor':
                D1_0 = (1. / r1) * (noise_s1 - noise_s)
                D1_1 = (1. / r2) * (noise_s2 - noise_s)
                D1 = (r2 * D1_0 - r1 * D1_1) / (r2 - r1)
                D2 = 2. * (D1_1 - D1_0) / (r2 - r1)
                x_t = (
                    (sigma_t / sigma_s)[(...,) + (None,)*dims] * x
                    - (alpha_t * phi_1)[(...,) + (None,)*dims] * noise_s
                    + (alpha_t * phi_2)[(...,) + (None,)*dims] * D1
                    - (alpha_t * phi_3)[(...,) + (None,)*dims] * D2
                )
            else:
                raise ValueError("solver_type must be either dpm_solver or dpm_solver++, got {}".format(solver_type))
        else:
            phi_11 = torch.expm1(r1 * h)
            phi_12 = torch.expm1(r2 * h)
            phi_1 = torch.expm1(h)
            phi_22 = torch.expm1(r2 * h) / (r2 * h) - 1.
            phi_2 = phi_1 / h - 1.
            phi_3 = phi_2 / h - 0.5

            if noise_s is None:
                noise_s = self.model_fn(x, s, panoptic=panoptic, mask_token=mask_token)
            if noise_s1 is None:
                x_s1 = (
                    torch.exp(log_alpha_s1 - log_alpha_s)[(...,) + (None,)*dims] * x
                    - (sigma_s1 * phi_11)[(...,) + (None,)*dims] * noise_s
                )
                noise_s1 = self.model_fn(x_s1, s1, panoptic=panoptic, mask_token=mask_token)
            if noise_s2 is None:
                x_s2 = (
                    torch.exp(log_alpha_s2 - log_alpha_s)[(...,) + (None,)*dims] * x
                    - (sigma_s2 * phi_12)[(...,) + (None,)*dims] * noise_s
                    - r2 / r1 * (sigma_s2 * phi_22)[(...,) + (None,)*dims] * (noise_s1 - noise_s)
                )
                noise_s2 = self.model_fn(x_s2, s2, panoptic=panoptic, mask_token=mask_token)
            if solver_type == 'dpm_solver':
                x_t = (
                    torch.exp(log_alpha_t - log_alpha_s)[(...,) + (None,)*dims] * x
                    - (sigma_t * phi_1)[(...,) + (None,)*dims] * noise_s
                    - (1. / r2) * (sigma_t * phi_2)[(...,) + (None,)*dims] * (noise_s2 - noise_s)
                )
            elif solver_type == 'taylor':
                D1_0 = (1. / r1) * (noise_s1 - noise_s)
                D1_1 = (1. / r2) * (noise_s2 - noise_s)
                D1 = (r2 * D1_0 - r1 * D1_1) / (r2 - r1)
                D2 = 2. * (D1_1 - D1_0) / (r2 - r1)
                x_t = (
                    torch.exp(log_alpha_t - log_alpha_s)[(...,) + (None,)*dims] * x
                    - (sigma_t * phi_1)[(...,) + (None,)*dims] * noise_s
                    - (sigma_t * phi_2)[(...,) + (None,)*dims] * D1
                    - (sigma_t * phi_3)[(...,) + (None,)*dims] * D2
                )
            else:
                raise ValueError("solver_type must be either dpm_solver or dpm_solver++, got {}".format(solver_type))

        if return_noise:
            return x_t, {'noise_s': noise_s, 'noise_s1': noise_s1, 'noise_s2': noise_s2}
        else:
            if enable_mask_opt==True:
                return x_t, pred_mask, mask_t #TODO: implement mask_t
            else:
                return x_t, pred_mask, pred_mask

    def dpm_solver_update(self, x, s, t, order, return_noise=False, solver_type='dpm_solver', r1=None, r2=None, panoptic=None, mask_token=None, enable_mask_opt=True,  use_ground_truth=False, enable_panoptic=False):
        """
        A single step for DPM-Solver of the given order `order`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (x.shape[0],).
            t: A pytorch tensor. The ending time, with the shape (x.shape[0],).
            order: A `int`. The order of DPM-Solver. We only support order == 1 or 2 or 3.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        if order == 1:
            return self.dpm_solver_first_update(x, s, t, return_noise=return_noise, panoptic=panoptic, mask_token=mask_token, enable_mask_opt=enable_mask_opt,  use_ground_truth=use_ground_truth, enable_panoptic=enable_panoptic)
        elif order == 2:
            return self.dpm_solver_second_update(x, s, t, return_noise=return_noise, solver_type=solver_type, r1=r1, panoptic=panoptic, mask_token=mask_token, enable_mask_opt=enable_mask_opt,  use_ground_truth=use_ground_truth, enable_panoptic=enable_panoptic)
        elif order == 3:
            return self.dpm_solver_third_update(x, s, t, return_noise=return_noise, solver_type=solver_type, r1=r1, r2=r2, panoptic=panoptic, mask_token=mask_token, enable_mask_opt=enable_mask_opt,  use_ground_truth=use_ground_truth, enable_panoptic=enable_panoptic)
        else:
            raise ValueError("Solver order must be 1 or 2 or 3, got {}".format(order))

    def dpm_multistep_update(self, x, noise_prev_list, t_prev_list, t, order, solver_type='taylor'):
        """
        A single step for DPM-Solver of the given order `order`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (x.shape[0],).
            t: A pytorch tensor. The ending time, with the shape (x.shape[0],).
            order: A `int`. The order of DPM-Solver. We only support order == 1 or 2 or 3.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        if order == 1:
            return self.dpm_solver_first_update(x, t_prev_list[-1], t, noise_s=noise_prev_list[-1])
        elif order == 2:
            return self.dpm_multistep_second_update(x, noise_prev_list, t_prev_list, t, solver_type=solver_type)
        elif order == 3:
            return self.dpm_multistep_third_update(x, noise_prev_list, t_prev_list, t, solver_type=solver_type)
        else:
            raise ValueError("Solver order must be 1 or 2 or 3, got {}".format(order))

    def dpm_solver_adaptive(self, x, order, t_T, t_0, h_init=0.05, atol=0.0078, rtol=0.05, theta=0.9, t_err=1e-5, solver_type='dpm_solver'):
        """
        The adaptive step size solver based on DPM-Solver.

        Args:
            x: A pytorch tensor. The initial value at time `t_T`.
            order: A `int`. The (higher) order of the solver. We only support order == 2 or 3.
            t_T: A `float`. The starting time of the sampling (default is T).
            t_0: A `float`. The ending time of the sampling (default is epsilon).
            h_init: A `float`. The initial step size (for logSNR).
            atol: A `float`. The absolute tolerance of the solver. For image data, the default setting is 0.0078, followed [1].
            rtol: A `float`. The relative tolerance of the solver. The default setting is 0.05.
            theta: A `float`. The safety hyperparameter for adapting the step size. The default setting is 0.9, followed [1].
            t_err: A `float`. The tolerance for the time. We solve the diffusion ODE until the absolute error between the 
                current time and `t_0` is less than `t_err`. The default setting is 1e-5.
        Returns:
            x_0: A pytorch tensor. The approximated solution at time `t_0`.

        [1] A. Jolicoeur-Martineau, K. Li, R. Piché-Taillefer, T. Kachman, and I. Mitliagkas, "Gotta go fast when generating data with score-based models," arXiv preprint arXiv:2105.14080, 2021.
        """
        ns = self.noise_schedule
        s = t_T * torch.ones((x.shape[0],)).to(x)
        lambda_s = ns.marginal_lambda(s)
        lambda_0 = ns.marginal_lambda(t_0 * torch.ones_like(s).to(x))
        h = h_init * torch.ones_like(s).to(x)
        x_prev = x
        nfe = 0
        if order == 2:
            r1 = 0.5
            lower_update = lambda x, s, t: self.dpm_solver_first_update(x, s, t, return_noise=True)
            higher_update = lambda x, s, t, **kwargs: self.dpm_solver_second_update(x, s, t, r1=r1, solver_type=solver_type, **kwargs)
        elif order == 3:
            r1, r2 = 1. / 3., 2. / 3.
            lower_update = lambda x, s, t: self.dpm_solver_second_update(x, s, t, r1=r1, return_noise=True, solver_type=solver_type)
            higher_update = lambda x, s, t, **kwargs: self.dpm_solver_third_update(x, s, t, r1=r1, r2=r2, solver_type=solver_type, **kwargs)
        else:
            raise ValueError("For adaptive step size solver, order must be 2 or 3, got {}".format(order))
        while torch.abs((s - t_0)).mean() > t_err:
            t = ns.inverse_lambda(lambda_s + h)
            x_lower, lower_noise_kwargs = lower_update(x, s, t)
            x_higher = higher_update(x, s, t, **lower_noise_kwargs)
            delta = torch.max(torch.ones_like(x).to(x) * atol, rtol * torch.max(torch.abs(x_lower), torch.abs(x_prev)))
            norm_fn = lambda v: torch.sqrt(torch.square(v.reshape((v.shape[0], -1))).mean(dim=-1, keepdim=True))
            E = norm_fn((x_higher - x_lower) / delta).max()
            if torch.all(E <= 1.):
                x = x_higher
                s = t
                x_prev = x_lower
                lambda_s = ns.marginal_lambda(s)
            h = torch.min(theta * h * torch.float_power(E, -1. / order).float(), lambda_0 - lambda_s)
            nfe += order
        print('adaptive solver nfe', nfe)
        return x

    def sample(self, x, steps=10, eps=1e-4, T=None, order=3, panoptic=None, skip_type='time_uniform',
        denoise=False, method='fast', solver_type='dpm_solver', atol=0.0078,
        rtol=0.05, mask_token=None, use_twophases=False, use_ground_truth=False, enable_panoptic=False, enable_mask_opt=False
    ):
        """
        Compute the sample at time `eps` by DPM-Solver, given the initial `x` at time `T`.

        We support the following algorithms:

            - Adaptive step size DPM-Solver (i.e. DPM-Solver-12 and DPM-Solver-23)

            - Fixed order DPM-Solver (i.e. DPM-Solver-1, DPM-Solver-2 and DPM-Solver-3).

            - Fast version of DPM-Solver (i.e. DPM-Solver-fast), which uses uniform logSNR steps and combine
                different orders of DPM-Solver. 

        **We recommend DPM-Solver-fast for both fast sampling in few steps (<=20) and fast convergence in many steps (50 to 100).**

        Choosing the algorithms:

            - If `adaptive_step_size` is True:
                We ignore `steps` and use adaptive step size DPM-Solver with a higher order of `order`.
                If `order`=2, we use DPM-Solver-12 which combines DPM-Solver-1 and DPM-Solver-2.
                If `order`=3, we use DPM-Solver-23 which combines DPM-Solver-2 and DPM-Solver-3.
                You can adjust the absolute tolerance `atol` and the relative tolerance `rtol` to balance the computatation costs
                (NFE) and the sample quality.

            - If `adaptive_step_size` is False and `fast_version` is True:
                We ignore `order` and use DPM-Solver-fast with number of function evaluations (NFE) = `steps`.
                We ignore `skip_type` and use uniform logSNR steps for DPM-Solver-fast.
                Given a fixed NFE=`steps`, the sampling procedure by DPM-Solver-fast is:
                    - Denote K = (steps // 3 + 1). We take K intermediate time steps for sampling.
                    - If steps % 3 == 0, we use (K - 2) steps of DPM-Solver-3, and 1 step of DPM-Solver-2 and 1 step of DPM-Solver-1.
                    - If steps % 3 == 1, we use (K - 1) steps of DPM-Solver-3 and 1 step of DPM-Solver-1.
                    - If steps % 3 == 2, we use (K - 1) steps of DPM-Solver-3 and 1 step of DPM-Solver-2.

            - If `adaptive_step_size` is False and `fast_version` is False:
                We use DPM-Solver-`order` for `order`=1 or 2 or 3, with total [`steps` // `order`] * `order` NFE.
                We support three types of `skip_type`:
                    - 'logSNR': uniform logSNR for the time steps, **recommended for DPM-Solver**.
                    - 'time_uniform': uniform time for the time steps. (Used in DDIM and DDPM.)
                    - 'time_quadratic': quadratic time for the time steps. (Used in DDIM.)

        =====================================================
        Args:
            x: A pytorch tensor. The initial value at time `T` (a sample from the normal distribution).
            steps: A `int`. The total number of function evaluations (NFE).
            eps: A `float`. The ending time of the sampling.
                We recommend `eps`=1e-3 when `steps` <= 15; and `eps`=1e-4 when `steps` > 15.
            T: A `float`. The starting time of the sampling. Default is `None`.
                If `T` is None, we use self.noise_schedule.T.
            order: A `int`. The order of DPM-Solver.
            skip_type: A `str`. The type for the spacing of the time steps. Default is 'logSNR'.
            adaptive_step_size: A `bool`. If true, use the adaptive step size DPM-Solver.
            fast_version: A `bool`. If true, use DPM-Solver-fast (recommended).
            atol: A `float`. The absolute tolerance of the adaptive step size solver.
            rtol: A `float`. The relative tolerance of the adaptive step size solver.
        Returns:
            x_0: A pytorch tensor. The approximated solution at time `t_0`.

        [1] A. Jolicoeur-Martineau, K. Li, R. Piché-Taillefer, T. Kachman, and I. Mitliagkas, "Gotta go fast when generating data with score-based models," arXiv preprint arXiv:2105.14080, 2021.
        """
        t_0 = eps
        t_T = self.noise_schedule.T if T is None else T
        device = x.device
        if method == 'adaptive':
            with torch.no_grad():
                x = self.dpm_solver_adaptive(x, order=order, t_T=t_T, t_0=t_0, atol=atol, rtol=rtol, solver_type=solver_type)
        elif method == 'multistep':
            assert steps >= order
            if timesteps is None:
                timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device=device)
            assert timesteps.shape[0] - 1 == steps
            with torch.no_grad():
                vec_t = timesteps[0].expand((x.shape[0]))
                noise_prev_list = [self.model_fn(x, vec_t)]
                t_prev_list = [vec_t]
                for init_order in range(1, order):
                    vec_t = timesteps[init_order].expand(x.shape[0])
                    x = self.dpm_multistep_update(x, noise_prev_list, t_prev_list, vec_t, init_order, solver_type=solver_type)
                    noise_prev_list.append(self.model_fn(x, vec_t))
                    t_prev_list.append(vec_t)
                for step in range(order, steps + 1):
                    vec_t = timesteps[step].expand(x.shape[0])
                    x = self.dpm_multistep_update(x, noise_prev_list, t_prev_list, vec_t, order, solver_type=solver_type)
                    for i in range(order - 1):
                        t_prev_list[i] = t_prev_list[i + 1]
                        noise_prev_list[i] = noise_prev_list[i + 1]
                    t_prev_list[-1] = vec_t
                    if step < steps:
                        noise_prev_list[-1] = self.model_fn(x, vec_t)
        elif method == 'fast':
            orders, _ = self.get_time_steps_for_dpm_solver_fast(skip_type=skip_type, t_T=t_T, t_0=t_0, steps=steps, order=order, device=device)
            timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device=device)
            with torch.no_grad():
                i = 0
                if mask_token is not None:
                    pred_mask = mask_token.to(x.device)
                    mask_t=mask_token.to(x.device)
                else:
                    pred_mask = mask_token
                    mask_t = mask_token
                #if mask_t is None:
                #    print("*****ERROR: mask token is none at beginning")
                for order in orders:
                    vec_s, vec_t = torch.ones((x.shape[0],)).to(device) * timesteps[i], torch.ones((x.shape[0],)).to(device) * timesteps[i + order]
                    h = self.noise_schedule.marginal_lambda(timesteps[i + order]) - self.noise_schedule.marginal_lambda(timesteps[i])
                    r1 = None if order <= 1 else (self.noise_schedule.marginal_lambda(timesteps[i + 1]) - self.noise_schedule.marginal_lambda(timesteps[i])) / h
                    r2 = None if order <= 2 else (self.noise_schedule.marginal_lambda(timesteps[i + 2]) - self.noise_schedule.marginal_lambda(timesteps[i])) / h
                    #optimize mask
                    #if mask_t is None:
                    #    print("*****ERROR: mask t becomes none at order ", i)
                    
                    x, pred_mask, mask_t = self.dpm_solver_update(x, vec_s, vec_t, order, solver_type=solver_type, r1=r1, r2=r2, panoptic=pred_mask, mask_token=mask_t, enable_mask_opt=enable_mask_opt, use_ground_truth=use_ground_truth, enable_panoptic=enable_panoptic)
                    #TODO:ground-truth mask with thrid order
                    #x, pred_mask, mask_t = self.dpm_solver_update(x, vec_s, vec_t, order, solver_type=solver_type, r1=r1, r2=r2, panoptic=panoptic, mask_token=mask_t, enable_mask_opt=False,  use_ground_truth=True, enable_panoptic=True)
                    i += order
            return x, pred_mask
        elif method == 'singlestep':
            N_steps = steps // order
            orders = [order,] * N_steps
            timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=N_steps, device=device)
            assert len(timesteps) - 1 == N_steps
            #NOTE:iteratively generate the mask by steps
            if mask_token is not None:
                mask_t = mask_token.to(x.device)
                pred_mask = mask_token.to(x.device)
            else:
                mask_t = mask_token
                pred_mask = mask_token
            with torch.no_grad():
                for i, order in enumerate(orders):
                    vec_s, vec_t = torch.ones((x.shape[0],)).to(device) * timesteps[i], torch.ones((x.shape[0],)).to(device) * timesteps[i + 1]
                    ##test starting from initial noise as mask queries at every step
                    #x, pred_mask, mask_t = self.dpm_solver_update(x, vec_s, vec_t, order, solver_type=solver_type,panoptic=panoptic, mask_token=mask_token)
                    #if use_twophases==False or i<N_steps/2:#phase one
                    #NOTE: input both pred_mask and mask_t 
                    x, pred_mask, mask_t = self.dpm_solver_update(x, vec_s, vec_t, order, solver_type=solver_type,panoptic=pred_mask, mask_token=mask_t,enable_mask_opt=enable_mask_opt, use_ground_truth=use_ground_truth, enable_panoptic=enable_panoptic)
                    #NOTE: ground-truth
                    #x, pred_mask, mask_t = self.dpm_solver_update(x, vec_s, vec_t, order, solver_type=solver_type,panoptic=panoptic, mask_token=mask_t,enable_mask_opt=False, use_ground_truth=True, enable_panoptic=True)
                    #baseline
                    #x, pred_mask, mask_t = self.dpm_solver_update(x, vec_s, vec_t, order, solver_type=solver_type,panoptic=panoptic, mask_token=mask_t,enable_mask_opt=False, use_ground_truth=False, enable_panoptic=False)
                    #else: #phase two, disable mask update, use mask_t from phase one
                    #    x, pred_mask_feat, mask_feat = self.dpm_solver_update(x, vec_s, vec_t, order, solver_type=solver_type,panoptic=panoptic, mask_token=mask_t, enable_mask_opt=False, use_ground_truth=True, enable_panoptic=True)
                if use_twophases==True:
                    for i, order in enumerate(orders):
                        vec_s, vec_t = torch.ones((x.shape[0],)).to(device) * timesteps[i], torch.ones((x.shape[0],)).to(device) * timesteps[i + 1]
                        x, pred_mask_feat, mask_feat = self.dpm_solver_update(x, vec_s, vec_t, order, solver_type=solver_type,panoptic=panoptic, mask_token=mask_t, enable_mask_opt=False, use_ground_truth=True, enable_panoptic=True)
                
                #NOTE: for analog bits pred_mask size:[N,8,H,W], mask_t size:[N,8,H,W]
                
            return x, pred_mask #pred_mask size:[N,200,H,W], (if using analog bits: [N,8,H,W]). mask_t size:[N,1,H,W]
        if denoise:
            x = self.denoise_fn(x, torch.ones((x.shape[0],)).to(device) * t_0)
        return x
