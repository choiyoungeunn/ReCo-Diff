import sys
sys.path.append('..')

from wrappers.basic_wrapper import DiffusionSparseWrapper
from networks.diffunet import UnetResNetBlock
from utilities.metrics import compute_SSIM
from datasets.aapmmyo import CTTools
import torch
import torch.nn as nn
import torch.nn.functional as F

# Model selection guide:
# Use 'ColdDiffusion' for CvG-Diff.
# Use 'ColdDiffusionErrCFG' for ReCo-Diff.

def insert_interpolated_elements(view_list, k):
    if len(view_list) < 2 or k <= 0:
        return view_list.copy()

    new_list = []
    for i in range(len(view_list) - 1):
        current = view_list[i]
        next_val = view_list[i + 1]
        new_list.append(current)

        # Calculate step size
        step = (next_val - current) / (k + 1)

        # Generate and append interpolated values
        for j in range(1, k + 1):
            interpolated = current + step * j
            new_list.append(round(interpolated))

    # Add the last element from original list
    new_list.append(view_list[-1])
    return new_list

# CvG-Diff mode
class ColdDiffusion(DiffusionSparseWrapper):
    def __init__(self,
         opt,
         **wrapper_kwargs
     ):
        super().__init__(**wrapper_kwargs)
        self.opt = opt
        self.denoise_fn = UnetResNetBlock(
            in_channels= 1,
            ch= opt.unet_dim
        )

        # The fewer views, the stronger the sparse artifacts.
        self.view_list = [288, 234, 180, 126, 72, 54, 36, 18]

        self.num_timesteps = len(self.view_list)

        self.cttool = CTTools()

    def generate_sparse_and_gt_data(self, mu_ct, num_views):
        if self.opt.dist:
            sparse_mu, gt_mu = self.module.generate_sparse_and_full_ct(mu_ct, num_views= num_views)
        else:
            sparse_mu, gt_mu = self.generate_sparse_and_full_ct(mu_ct, num_views= num_views)

        return sparse_mu, gt_mu

    def forward(self, x, t):
        return self.denoise_fn(x, t)

    @torch.no_grad()
    def sample(self, x, t= None):
        self.denoise_fn.eval()
        if t == None:
            t = self.num_timesteps - 1

        b = x.shape[0]
        x_deg = x
        t_start = t
        t_id_list = [t_start, 0]

        sequential_budget = t_start + 1
        t_id_list = insert_interpolated_elements(t_id_list, sequential_budget - 2)
        for id_id, t_id in enumerate(t_id_list):
            step = torch.full((b, ), t_id, dtype=torch.long, device=x.device)
            x0_hat = self.denoise_fn(x_deg, step)
            if id_id == 0:
                direct_recon = x0_hat
            x_current_deg_estimate, _ = self.generate_sparse_and_full_ct(x0_hat, num_views= self.view_list[t_id])

            if id_id < len(t_id_list) - 1:
                x_next_deg_estimate, _ = self.generate_sparse_and_full_ct(x0_hat, num_views=self.view_list[t_id_list[id_id + 1]])
                x_deg = x_deg - x_current_deg_estimate + x_next_deg_estimate
            else:
                x_deg = x0_hat

        return x_deg, direct_recon
    
    # Semantic-Prioritized Dual-Phase Sampling (SPDPS)
    @torch.no_grad()
    def iterative_sample(self, x_deg, t_start, iterative_budget, refine_budget=2):
        self.denoise_fn.eval()
        x_deg_in = x_deg

        current_t = t_start
        b = x_deg.shape[0]

        previous_direct_recon = None
        direct_recon = None

        total_budget = 0

        # Semantic Correction
        while (iterative_budget > 0):
            total_budget += 1
            step = torch.full((b,), current_t, dtype=torch.long, device=x_deg.device)
            x0_hat = self.denoise_fn(x_deg, step)
            if direct_recon is None:
                direct_recon = x0_hat
            if previous_direct_recon is None:
                sequential_flag = True
            elif compute_SSIM(self.cttool.window_transform(self.cttool.mu2HU(x0_hat)),
                              self.cttool.window_transform(self.cttool.mu2HU(previous_direct_recon)), data_range=1,
                              spatial_dims=2) < self.opt.time_back_ssim_threshold:
                sequential_flag = True
                print('continue SSIM')
            else:
                sequential_flag = False
                iterative_budget = iterative_budget - 1
                print('reset SSIM')

            if iterative_budget == 0:
                x_current_deg_estimate, _ = self.generate_sparse_and_full_ct(x0_hat, num_views=self.view_list[current_t])
                x_next_deg_estimate, _ = self.generate_sparse_and_full_ct(x0_hat, num_views=self.view_list[refine_budget - 1])
                current_t = refine_budget - 1
                x_deg = x_deg - x_current_deg_estimate + x_next_deg_estimate
            else:
                if sequential_flag:
                    x_current_deg_estimate, _ = self.generate_sparse_and_full_ct(x0_hat, num_views=self.view_list[current_t])
                    if current_t > 0:
                        x_next_deg_estimate, _ = self.generate_sparse_and_full_ct(x0_hat, num_views=self.view_list[current_t - 1])
                    else:
                        x_next_deg_estimate = x0_hat
                    current_t = current_t - 1
                    x_deg = x_deg - x_current_deg_estimate + x_next_deg_estimate
                    previous_direct_recon = x0_hat
                else:
                    # If we find no further improvement, we go back to start time-step
                    x_current_deg_estimate, _ = self.generate_sparse_and_full_ct(x0_hat, num_views=self.view_list[t_start])
                    x_next_deg_estimate, _ = self.generate_sparse_and_full_ct(x0_hat, num_views=self.view_list[t_start - 1])
                    current_t = t_start - 1
                    x_deg = x_deg_in - x_current_deg_estimate + x_next_deg_estimate
                    previous_direct_recon = None

        # Detail Refinement
        for i in range(refine_budget):
            total_budget += 1
            step = torch.full((b,), current_t, dtype=torch.long, device=x_deg.device)
            x0_hat = self.denoise_fn(x_deg, step)
            x_current_deg_estimate, _ = self.generate_sparse_and_full_ct(x0_hat, num_views=self.view_list[current_t])
            if current_t > 0:
                x_next_deg_estimate, _ = self.generate_sparse_and_full_ct(x0_hat, num_views=self.view_list[current_t - 1])
                x_deg = x_deg - x_current_deg_estimate + x_next_deg_estimate
            else:
                x_deg = x0_hat

            current_t = current_t - 1
            
        return x_deg, direct_recon

# ReCo-Diff mode
# 'in_channels=2' is used for Residual-Conditioned Self-Guided Sampling (ReCo):
# one channel is the sparse input ('x_deg'), and the other is the residual/error condition ('err').
class ColdDiffusionErrCFG(ColdDiffusion):
    def __init__(self, opt, **wrapper_kwargs):
        super().__init__(opt, **wrapper_kwargs)
        self.denoise_fn = UnetResNetBlock(
            in_channels=2,
            ch=opt.unet_dim
        )

    def _normalize_err(self, err):
        sigma = getattr(self.opt, 'err_cfg_sigma', 0.0)
        if sigma and sigma > 0:
            err = torch.tanh(err / sigma)
        return err

    def _estimate_sparse(self, x0_hat, num_views):
        sparse_mu_hat, _ = self.generate_sparse_and_full_ct(x0_hat, num_views=num_views)
        return sparse_mu_hat

    def denoise_err_conditioned(self, x_deg, t, err):
        x_in = torch.cat([x_deg, err], dim=1)
        return self.denoise_fn(x_in, t)

    def denoise_err_cfg(self, x_deg, t, num_views):
        zero_err = torch.zeros_like(x_deg)
        x_in_uncond = torch.cat([x_deg, zero_err], dim=1)
        x0_uncond = self.denoise_fn(x_in_uncond, t)
        sparse_hat = self._estimate_sparse(x0_uncond, num_views)
        err = self._normalize_err(x_deg - sparse_hat)
        err_scalar = err.abs().mean(dim=(1, 2, 3), keepdim=True)
        x_in_cond = torch.cat([x_deg, err], dim=1)
        x0_cond = self.denoise_fn(x_in_cond, t)
        
        print(f"{err_scalar.item():10.4f}\t{num_views}")
        return x0_cond

    def _err_sched_beta(self, err, delta=None):
        mode = getattr(self.opt, 'err_cfg_sched_mode', 'err')
        tau = float(getattr(self.opt, 'err_cfg_sched_tau', 1.0))
        tau = max(tau, 1e-8)
        if mode == 'delta' and delta is not None:
            score = delta.abs().mean(dim=(1, 2, 3), keepdim=True)
        else:
            score = err.abs().mean(dim=(1, 2, 3), keepdim=True)
        return torch.clamp(score / tau, 0.0, 1.0)

    def denoise_err_scheduled(self, x_deg, t, num_views):
        zero_err = torch.zeros_like(x_deg)
        # err=0 target vs err-conditioned prediction, mixed by scheduling weight.
        x_in_target = torch.cat([x_deg, zero_err], dim=1)
        x0_target = self.denoise_fn(x_in_target, t)
        sparse_hat = self._estimate_sparse(x0_target, num_views)
        err = self._normalize_err(x_deg - sparse_hat)
        x_in_err = torch.cat([x_deg, err], dim=1)
        x0_err = self.denoise_fn(x_in_err, t)
        delta = x0_err - x0_target
        beta = self._err_sched_beta(err, delta=delta)
        x0_sched = (1.0 - beta) * x0_target + beta * x0_err
        return x0_sched, beta


    # Sampling used in colddiff_trainer.py
    @torch.no_grad()
    def sample_err_cfg(self, x, t=None):
        self.denoise_fn.eval()
        if t is None:
            t = self.num_timesteps - 1

        b = x.shape[0]
        x_deg = x
        t_start = t
        t_id_list = [t_start, 0]

        sequential_budget = t_start + 1
        t_id_list = insert_interpolated_elements(t_id_list, sequential_budget - 2)
        for id_id, t_id in enumerate(t_id_list):
            step = torch.full((b,), t_id, dtype=torch.long, device=x.device)
            num_views = self.view_list[t_id]
            x0_hat = self.denoise_err_cfg(x_deg, step, num_views)
            if id_id == 0:
                direct_recon = x0_hat
            x_current_deg_estimate, _ = self.generate_sparse_and_full_ct(x0_hat, num_views=num_views)

            if id_id < len(t_id_list) - 1:
                next_views = self.view_list[t_id_list[id_id + 1]]
                x_next_deg_estimate, _ = self.generate_sparse_and_full_ct(x0_hat, num_views=next_views)
                x_deg = x_deg - x_current_deg_estimate + x_next_deg_estimate
            else:
                x_deg = x0_hat
        
        return x_deg, direct_recon


    # Sampling used in colddiff_tester.py
    @torch.no_grad()
    def iterative_sample_err_cfg_reset_views(
        self, x_deg, t_start, refine_budget=2, 
    ):
        self.denoise_fn.eval()
        
        x_deg_in = x_deg
        current_t = t_start
        b = x_deg.shape[0]

        direct_recon = None
        reset_after_steps = 2
        step_count = 0
        reset_done = False

        while current_t >= refine_budget:
            step = torch.full((b,), current_t, dtype=torch.long, device=x_deg.device)
            num_views = self.view_list[current_t]
            x0_hat = self.denoise_err_cfg(x_deg, step, num_views)
            if direct_recon is None:
                direct_recon = x0_hat

            step_count += 1
            if not reset_done and step_count == reset_after_steps:
                if t_start > 0:
                    x_current_deg_estimate, _ = self.generate_sparse_and_full_ct(x0_hat, num_views=self.view_list[t_start])
                    x_next_deg_estimate, _ = self.generate_sparse_and_full_ct(x0_hat, num_views=self.view_list[t_start - 1])
                    x_deg = x_deg_in - x_current_deg_estimate + x_next_deg_estimate
                else:
                    x_deg = x0_hat
                reset_done = True
                current_t = t_start - 1
                continue

            x_current_deg_estimate, _ = self.generate_sparse_and_full_ct(x0_hat, num_views=self.view_list[current_t])
            if current_t > 0:
                x_next_deg_estimate, _ = self.generate_sparse_and_full_ct(x0_hat, num_views=self.view_list[current_t - 1])
            else:
                x_next_deg_estimate = x0_hat
            x_deg = x_deg - x_current_deg_estimate + x_next_deg_estimate
            current_t = current_t - 1

        refine_steps = min(refine_budget, current_t + 1)
        for _ in range(refine_steps):
            step = torch.full((b,), current_t, dtype=torch.long, device=x_deg.device)
            num_views = self.view_list[current_t]
            x0_hat = self.denoise_err_cfg(x_deg, step, num_views)
            x_current_deg_estimate, _ = self.generate_sparse_and_full_ct(x0_hat, num_views=self.view_list[current_t])
            if current_t > 0:
                x_next_deg_estimate, _ = self.generate_sparse_and_full_ct(x0_hat, num_views=self.view_list[current_t - 1])
                x_deg = x_deg - x_current_deg_estimate + x_next_deg_estimate
            else:
                x_deg = x0_hat

            current_t = current_t - 1

        return x_deg, direct_recon


    # One-Time Level Transition for Extremely Sparse Regimes.
    # 'print('reset SSIM')' is kept intentionally for sequence comparison with CvG-Diff.
    # Although 'iterative_sample_err_noSSIM' does not compute SSIM, this log marks the point where the process enters the Detail Refinement stage (same transition point as CvG-Diff).
    # Please refer to the graph in Figure 4.
    @torch.no_grad()
    def iterative_sample_err_noSSIM(self, x_deg, t_start, iterative_budget=2, refine_budget=5):
        self.denoise_fn.eval()

        x_deg_in = x_deg
        current_t = t_start
        b = x_deg.shape[0]

        previous_direct_recon = None
        direct_recon = None

        total_budget = 0

        while (iterative_budget > 0):
            total_budget += 1
            step = torch.full((b,), current_t, dtype=torch.long, device=x_deg.device)
            num_views = self.view_list[current_t]
            x0_hat = self.denoise_err_cfg(x_deg, step, num_views)
            if direct_recon is None:
                direct_recon = x0_hat
            if previous_direct_recon is None:
                sequential_flag = True
            else:
                sequential_flag = False
                iterative_budget = iterative_budget - 1
                print('reset SSIM')

            if iterative_budget == 0:
                x_current_deg_estimate, _ = self.generate_sparse_and_full_ct(x0_hat, num_views=self.view_list[current_t])
                x_next_deg_estimate, _ = self.generate_sparse_and_full_ct(x0_hat, num_views=self.view_list[refine_budget - 1])
                current_t = refine_budget - 1
                x_deg = x_deg - x_current_deg_estimate + x_next_deg_estimate
            else:
                if sequential_flag:
                    x_current_deg_estimate, _ = self.generate_sparse_and_full_ct(x0_hat, num_views=self.view_list[current_t])
                    if current_t > 0:
                        x_next_deg_estimate, _ = self.generate_sparse_and_full_ct(x0_hat, num_views=self.view_list[current_t - 1])
                    else:
                        x_next_deg_estimate = x0_hat
                    current_t = current_t - 1
                    x_deg = x_deg - x_current_deg_estimate + x_next_deg_estimate
                    previous_direct_recon = x0_hat
                else:
                    x_current_deg_estimate, _ = self.generate_sparse_and_full_ct(x0_hat, num_views=self.view_list[t_start])
                    x_next_deg_estimate, _ = self.generate_sparse_and_full_ct(x0_hat, num_views=self.view_list[t_start - 1])
                    current_t = t_start - 1
                    x_deg = x_deg_in - x_current_deg_estimate + x_next_deg_estimate
                    previous_direct_recon = None

        for i in range(refine_budget):
            total_budget += 1
            step = torch.full((b,), current_t, dtype=torch.long, device=x_deg.device)
            num_views = self.view_list[current_t]
            x0_hat = self.denoise_err_cfg(x_deg, step, num_views)
            x_current_deg_estimate, _ = self.generate_sparse_and_full_ct(x0_hat, num_views=self.view_list[current_t])
            if current_t > 0:
                x_next_deg_estimate, _ = self.generate_sparse_and_full_ct(x0_hat, num_views=self.view_list[current_t - 1])
                x_deg = x_deg - x_current_deg_estimate + x_next_deg_estimate
            else:
                x_deg = x0_hat

            current_t = current_t - 1

        print('-------------')
        return x_deg, direct_recon