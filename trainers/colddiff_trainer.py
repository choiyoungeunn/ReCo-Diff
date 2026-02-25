import os
import wandb
import tqdm
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import sys

sys.path.append('..')
from trainers.basic_trainer import BasicTrainer
from datasets.aapmmyo import AAPMMyoDataset

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class ColdDiffTrainer(BasicTrainer):
    def __init__(self, opt=None, net=None, loss_type='l2'):
        super().__init__()
        self.opt = opt
        self.net = net

        dataset_name = opt.dataset_name.lower()
        if dataset_name == 'aapm':
            self.train_dataset = AAPMMyoDataset(opt.dataset_path, mode='train', dataset_shape=opt.dataset_shape)
            self.val_dataset = AAPMMyoDataset(opt.dataset_path, mode='val', dataset_shape=opt.dataset_shape)
        else:
            raise NotImplementedError(f'Dataset {dataset_name} not implemented, try aapm.')
        self.checkpoint_path = os.path.join(opt.checkpoint_root, opt.checkpoint_dir)

        if opt.use_wandb:
            if opt.local_rank == 0:  # only on main process
                self.wandb_init(self.opt)

        self.tb_writer = None
        if getattr(opt, 'use_tensorboard', False) and opt.local_rank == 0:
            tb_root = getattr(opt, 'tensorboard_root', '')
            tb_dir = getattr(opt, 'tensorboard_dir', '')
            if tb_root == '' and tb_dir == '':
                tb_dir = os.path.join(opt.checkpoint_root, opt.checkpoint_dir, 'tensorboard')
            tb_path = os.path.join(tb_root, tb_dir) if tb_root != '' else tb_dir
            if tb_path == '':
                tb_path = os.path.join(opt.checkpoint_dir, 'tensorboard')
            os.makedirs(tb_path, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=tb_path)

        self.best_val_loss = np.inf

        self.update_ema_iter = opt.update_ema_iter
        self.start_ema_iter = opt.start_ema_iter
        self.ema_net = copy.deepcopy(net)
        # set grad false
        for param in self.ema_net.parameters():
            param.requires_grad = False
        self.ema = EMA(opt.ema_decay)

        self.criterion = self.get_pixel_criterion(loss_type)
        self.itlog_intv = opt.log_interval

    def _tb_prepare_image(self, mu, width=3000, center=500):
        if mu.dim() == 4:
            mu = mu[0]
        hu = self.cttool.window_transform(self.cttool.mu2HU(mu), width=width, center=center)
        return hu.detach().float().cpu().clamp(0.0, 1.0)

    def _tb_prepare_err(self, err):
        if err.dim() == 4:
            err = err[0]
        err = err.detach().float().cpu()
        err_min = err.min()
        err_max = err.max()
        if (err_max - err_min) > 0:
            err = (err - err_min) / (err_max - err_min)
        else:
            err = err * 0.0
        return err.clamp(0.0, 1.0)

    def _tb_log_scalars(self, rel_path, step, **kwargs):
        if self.tb_writer is None:
            return
        self.tensorboard_scalar(self.tb_writer, rel_path, step, **kwargs)

    def _tb_log_images(self, rel_path, step, **kwargs):
        if self.tb_writer is None:
            return
        for key, value in kwargs.items():
            self.tb_writer.add_image(
                os.path.join(rel_path, key),
                value,
                global_step=step,
                dataformats='CHW',
            )

    @staticmethod
    def wandb_init(opt, key=None):
        if key is None:
            print('WANDB key not provided, attempting anonymous login...')
        else:
            wandb.login(key=key)
        wandb_root = opt.wandb_root
        wandb_dir = opt.wandb_dir
        wandb_path = os.path.join(wandb_root, wandb_dir)
        if not os.path.exists(wandb_path):
            os.makedirs(wandb_path)
        # wandb.init(project=opt.wandb_project, name=str(wandb_dir), dir=wandb_path, resume='allow', reinit=True,)
        wandb.init(project=opt.wandb_project, entity=opt.wandb_entity, name=opt.run_name, config=opt)

    def save_opt(self, optimizer=None, scheduler= None, opt_name=''):
        checkpoint_path = os.path.join(self.opt.checkpoint_root, self.opt.checkpoint_dir)
        optimizer_param = optimizer.state_dict() if optimizer is not None else self.optimizer.state_dict()
        scheduler_param = scheduler.state_dict()
        if scheduler is not None:
            opt_check = {
                'optimizer': optimizer_param,
                'scheduler': scheduler_param,
                'epoch': self.epoch,
            }
        else:
            opt_check = {
                'optimizer': optimizer_param,
                'epoch' : self.epoch,
            }
        self.save_checkpoint(opt_check, checkpoint_path, self.opt.checkpoint_dir +'-opt-' + opt_name, 'latest')

    def reset_parameters(self):
        if self.opt.dist:
            self.ema_net.load_state_dict(self.net.module.state_dict())
        else:
            self.ema_net.load_state_dict(self.net.state_dict())


    def step_ema(self, n_iter):
        if n_iter < self.start_ema_iter:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_net, self.net)

    def generate_sparse_and_gt_data(self, mu_ct, num_views):
        if self.opt.dist:
            sparse_mu, gt_mu = self.net.module.generate_sparse_and_full_ct(mu_ct, num_views= num_views)
        else:
            sparse_mu, gt_mu = self.net.generate_sparse_and_full_ct(mu_ct, num_views= num_views)

        return sparse_mu, gt_mu

    def _err_cfg_normalize(self, err):
        sigma = getattr(self.opt, 'err_cfg_sigma', 0.0)
        if sigma and sigma > 0:
            err = torch.tanh(err / sigma)
        return err

    def _err_cfg_sparse_hat(self, recon_mu, num_views):
        sparse_mu_hat, _ = self.generate_sparse_and_gt_data(recon_mu, num_views=num_views)
        return sparse_mu_hat

    def _err_cfg_base_recon(self, model, x_deg, t):
        zero_err = torch.zeros_like(x_deg)
        x_in = torch.cat([x_deg, zero_err], dim=1)
        return model(x_in, t)

    # ReCo: Residual-Conditioned Self-Guided Sampling
    def _err_cfg_forward(self, model, x_deg, t, num_views):
        with torch.no_grad():
            base_recon = self._err_cfg_base_recon(model, x_deg, t)
            err = x_deg - self._err_cfg_sparse_hat(base_recon, num_views)
            err = self._err_cfg_normalize(err)
        x_in = torch.cat([x_deg, err], dim=1)
        return model(x_in, t), err

    def train(self, ):
        self.iter_log_flag = False
        losses, rmses, psnrs, ssims = [], [], [], []
        err_scalars, err_nxt_scalars = [], []

        self.net.train()
        self.ema_net.train()
        pbar = tqdm.tqdm(self.train_loader, ncols=60) if self.opt.use_tqdm else self.train_loader
        for i, data in enumerate(pbar):
            mu_ct = data.to('cuda')
            b = mu_ct.shape[0]
            if self.opt.dist:
                t_single = torch.randint(0, self.net.module.num_timesteps, (1,), device=mu_ct.device).long()
            else:
                t_single = torch.randint(0, self.net.num_timesteps, (1, ), device=mu_ct.device).long()

            t = t_single.repeat((b, ))
            if self.opt.dist:
                num_views = self.net.module.view_list[t_single.item()]
            else:
                num_views = self.net.view_list[t_single.item()]
            sparse_mu, gt_mu = self.generate_sparse_and_gt_data(mu_ct, num_views= num_views)
            if self.opt.use_err_cfg:
                recon_mu, err = self._err_cfg_forward(self.net, sparse_mu, t, num_views)
            else:
                recon_mu = self.net(sparse_mu, t)
                err = None
            # restore loss
            loss = self.criterion(recon_mu, gt_mu)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.optimizer.step()

            # calculate the accuracy
            losses.append(loss.item())
            rmse, psnr, ssim = self.get_metrics_by_window(recon_mu, gt_mu)
            rmses.append(rmse)
            psnrs.append(psnr)
            ssims.append(ssim)
            if err is not None:
                err_scalars.append(err.abs().mean().item())

            # log acc by iteration
            if self.opt.local_rank == 0:
                if self.iter != 0 and self.iter % self.itlog_intv == 0:
                    log_info = {
                        'loss': np.mean(losses[-self.itlog_intv:]),
                        'rmse': np.mean(rmses[-self.itlog_intv:]),
                        'ssim': np.mean(ssims[-self.itlog_intv:]),
                        'psnr': np.mean(psnrs[-self.itlog_intv:]), }

                    if self.opt.use_wandb:
                        self.wandb_logger('train/iter', **log_info)
                    self._tb_log_scalars('train/iter', self.iter, **log_info)
                    if self.tb_writer is not None and err is not None:
                        self._tb_log_images(
                            'train/images',
                            self.iter,
                            err=self._tb_prepare_err(err),
                        )
                        err_scalar = err.abs().mean().item()
                        self._tb_log_scalars('train/iter', self.iter, **{'err_scalar': err_scalar})
                    if self.tb_writer is not None:
                        current_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                        self._tb_log_scalars('train/iter', self.iter, **{'lr': current_lr})
            self.iter += 1

            # ema update
            if self.iter % self.update_ema_iter == 0:
                self.step_ema(self.iter)

            if self.iter > self.start_ema_iter:
                if self.opt.use_err_cfg:
                    recon_mu, _ = self._err_cfg_forward(self.ema_net, sparse_mu, t, num_views)
                    recon_mu = recon_mu.detach()
                else:
                    recon_mu = self.ema_net(sparse_mu, t).detach()
                err_nxt = None
                if t_single.item() > 0:
                    t_nxt = torch.randint(0, t_single.item(), (1,), device=mu_ct.device).long()
                    t_new = t_nxt.repeat((b,))
                    if self.opt.dist:
                        num_views_nxt = self.net.module.view_list[t_nxt.item()]
                    else:
                        num_views_nxt = self.net.view_list[t_nxt.item()]
                    current_sparse_mu_hat, _ = self.generate_sparse_and_gt_data(recon_mu, num_views=num_views)
                    nxt_sparse_mu_hat, _ = self.generate_sparse_and_gt_data(recon_mu, num_views=num_views_nxt)

                    nxt_sparse_mu = sparse_mu - current_sparse_mu_hat + nxt_sparse_mu_hat
                    if self.opt.use_err_cfg:
                        nxt_recon_mu, err_nxt = self._err_cfg_forward(self.net, nxt_sparse_mu, t_new, num_views_nxt)
                    else:
                        nxt_recon_mu = self.net(nxt_sparse_mu, t_new)
                else:
                    if self.opt.use_err_cfg:
                        zero_err = torch.zeros_like(recon_mu)
                        x_in = torch.cat([recon_mu, zero_err], dim=1)
                        nxt_recon_mu = self.net(x_in, t)
                    else:
                        nxt_recon_mu = self.net(recon_mu, t)

                # compose loss (EPCT)
                loss = self.criterion(nxt_recon_mu, gt_mu)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.optimizer.step()

                # calculate the accuracy
                losses.append(loss.item())
                rmse, psnr, ssim = self.get_metrics_by_window(recon_mu, gt_mu)
                rmses.append(rmse)
                psnrs.append(psnr)
                ssims.append(ssim)
                if err_nxt is not None:
                    err_nxt_scalars.append(err_nxt.abs().mean().item())

                # log acc by iteration
                if self.opt.local_rank == 0:
                    if self.iter != 0 and self.iter % self.itlog_intv == 0:
                        log_info = {
                            'loss': np.mean(losses[-self.itlog_intv:]),
                            'rmse': np.mean(rmses[-self.itlog_intv:]),
                            'ssim': np.mean(ssims[-self.itlog_intv:]),
                            'psnr': np.mean(psnrs[-self.itlog_intv:]), }

                        if self.opt.use_wandb:
                            self.wandb_logger('train/iter', **log_info)
                        self._tb_log_scalars('train/iter', self.iter, **log_info)
                        if self.tb_writer is not None and err_nxt is not None:
                            self._tb_log_images(
                                'train/images',
                                self.iter,
                                err_next=self._tb_prepare_err(err_nxt),
                            )
                            err_scalar_nxt = err_nxt.abs().mean().item()
                            self._tb_log_scalars('train/iter', self.iter, **{'err_scalar_nxt': err_scalar_nxt})
                self.iter += 1

                if self.iter % self.update_ema_iter == 0:
                    self.step_ema(self.iter)

        # epoch info
        if self.opt.local_rank == 0:
            print('Logging epoch information...')
            epoch_log = {
                'loss': np.mean(losses),
                'rmse': np.mean(rmses),
                'ssim': np.mean(ssims),
                'psnr': np.mean(psnrs),}
            if err_scalars:
                epoch_log['err_scalar'] = np.mean(err_scalars)
            if err_nxt_scalars:
                epoch_log['err_scalar_nxt'] = np.mean(err_nxt_scalars)
            current_lr = self.optimizer.state_dict()['param_groups'][0]['lr']

            print(f'Epoch {self.epoch} learning rate: {current_lr}')
            print(f'Epoch {self.epoch} train loss: {epoch_log["loss"]}')
            print(f'Epoch {self.epoch} train rmse: {epoch_log["rmse"]}')
            print(f'Epoch {self.epoch} train ssim: {epoch_log["ssim"]}')
            print(f'Epoch {self.epoch} train psnr: {epoch_log["psnr"]}')

            if self.opt.use_wandb:
                self.wandb_logger('train/epoch', step_name='epoch', step=self.epoch, **epoch_log)
                self.wandb_logger('settings', step_name='epoch', step=self.epoch,
                                  **{'current_lr': current_lr, 'batch_size': self.opt.batch_size})
            self._tb_log_scalars('train/epoch', self.epoch, **epoch_log)
            self._tb_log_scalars('settings', self.epoch, **{'current_lr': current_lr, 'batch_size': self.opt.batch_size})

    def val(self, ):
        self.net.eval()
        self.ema_net.eval()
        losses = []
        rmses, psnrs, ssims = [], [], []
        rmses_direct, psnrs_direct, ssims_direct = [], [], []
        ema_rmses, ema_psnrs, ema_ssims = [], [], []
        ema_rmses_direct, ema_psnrs_direct, ema_ssims_direct = [], [], []

        pbar = tqdm.tqdm(self.val_loader, ncols=60) if self.opt.use_tqdm else self.val_loader
        with torch.no_grad():
            for i, data in enumerate(pbar):
                mu_ct = data.to('cuda')
                if self.opt.dist:
                    num_views = self.net.module.view_list[-1]
                    timestep_st = self.net.module.num_timesteps - 1
                else:
                    num_views = self.net.view_list[-1]
                    timestep_st = self.net.num_timesteps - 1
                sparse_mu, gt_mu = self.generate_sparse_and_gt_data(mu_ct, num_views= num_views)
                if self.opt.use_err_cfg:
                    if self.opt.dist:
                        recon_mu, direct_recon_mu = self.net.module.sample_err_cfg(sparse_mu, timestep_st)
                    else:
                        recon_mu, direct_recon_mu = self.net.sample_err_cfg(sparse_mu, timestep_st)
                    recon_mu_ema, direct_recon_mu_ema = self.ema_net.sample_err_cfg(sparse_mu, timestep_st)
                else:
                    if self.opt.dist:
                        recon_mu, direct_recon_mu = self.net.module.sample(sparse_mu, timestep_st)
                    else:
                        recon_mu, direct_recon_mu = self.net.sample(sparse_mu, timestep_st)
                    recon_mu_ema, direct_recon_mu_ema = self.ema_net.sample(sparse_mu, timestep_st)

                # restore loss
                loss = self.criterion(recon_mu, gt_mu)
                losses.append(loss.item())

                rmse, psnr, ssim = self.get_metrics_by_window(recon_mu, gt_mu)
                rmses.append(rmse)
                psnrs.append(psnr)
                ssims.append(ssim)

                rmse_direct, psnr_direct, ssim_direct = self.get_metrics_by_window(direct_recon_mu, gt_mu)
                rmses_direct.append(rmse_direct)
                psnrs_direct.append(psnr_direct)
                ssims_direct.append(ssim_direct)

                ema_rmse, ema_psnr, ema_ssim = self.get_metrics_by_window(recon_mu_ema, gt_mu)
                ema_rmses.append(ema_rmse)
                ema_psnrs.append(ema_psnr)
                ema_ssims.append(ema_ssim)

                ema_rmse_direct, ema_psnr_direct, ema_ssim_direct = self.get_metrics_by_window(direct_recon_mu_ema, gt_mu)
                ema_rmses_direct.append(ema_rmse_direct)
                ema_psnrs_direct.append(ema_psnr_direct)
                ema_ssims_direct.append(ema_ssim_direct)

                if self.opt.local_rank == 0 and self.tb_writer is not None and i == 0:
                    self._tb_log_images(
                        'val/images',
                        self.epoch,
                        sparse_mu=self._tb_prepare_image(sparse_mu),
                        recon_mu=self._tb_prepare_image(recon_mu),
                        direct_recon_mu=self._tb_prepare_image(direct_recon_mu),
                        recon_mu_ema=self._tb_prepare_image(recon_mu_ema),
                        direct_recon_mu_ema=self._tb_prepare_image(direct_recon_mu_ema),
                        gt_mu=self._tb_prepare_image(gt_mu),
                    )

        save_condition = False

        if np.mean(losses) < self.best_val_loss:
            self.best_val_loss = np.mean(losses)
            save_condition = True

        if self.opt.local_rank == 0:
            print('Logging validation information...')
            epoch_log = {
                'loss': np.mean(losses),
                'rmse': np.mean(rmses), 'rmse_direct': np.mean(rmses_direct), 'ema_rmse': np.mean(ema_rmses), 'ema_rmse_direct': np.mean(ema_rmses_direct),
                'ssim': np.mean(ssims), 'ssim_direct': np.mean(ssims_direct), 'ema_ssim': np.mean(ema_ssims), 'ema_ssim_direct': np.mean(ema_ssims_direct),
                'psnr': np.mean(psnrs), 'psnr_direct': np.mean(psnrs_direct), 'ema_psnr': np.mean(ema_psnrs), 'ema_psnr_direct': np.mean(ema_psnrs_direct), }

            print(f'Epoch {self.epoch} val loss: {epoch_log["loss"]}')
            print(f'Epoch {self.epoch} val rmse, multi-step: {epoch_log["rmse"]}, direct: {epoch_log["rmse_direct"]}')
            print(f'Epoch {self.epoch} val ssim, multi-step: {epoch_log["ssim"]}, direct: {epoch_log["ssim_direct"]}')
            print(f'Epoch {self.epoch} val psnr, multi-step: {epoch_log["psnr"]}, direct: {epoch_log["psnr_direct"]}')
            print(f'Epoch {self.epoch} val ema rmse, multi-step: {epoch_log["ema_rmse"]}, direct: {epoch_log["ema_rmse_direct"]}')
            print(f'Epoch {self.epoch} val ema ssim, multi-step: {epoch_log["ema_ssim"]}, direct: {epoch_log["ema_ssim_direct"]}')
            print(f'Epoch {self.epoch} val ema psnr, multi-step: {epoch_log["ema_psnr"]}, direct: {epoch_log["ema_psnr_direct"]}')


            if self.opt.use_wandb:
                self.wandb_logger('val/epoch', step_name='epoch', step=self.epoch, **epoch_log)
            self._tb_log_scalars('val/epoch', self.epoch, **epoch_log)

        return save_condition

    def fit(self, ):
        opt = self.opt
        if opt.dist:
            opt.local_rank = int(os.environ["LOCAL_RANK"])
            self.net = nn.SyncBatchNorm.convert_sync_batchnorm(self.net)
            self.ema_net = nn.SyncBatchNorm.convert_sync_batchnorm(self.ema_net)
            dist.init_process_group(backend='nccl')

        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)

        print(f'''Summary:
            Sparse Angles:          {opt.num_views}
            Number of Epochs:    {opt.epochs}
            Batch Size:                {opt.batch_size}
            Initial Learning rate:   {opt.lr}
            Training Size:             {len(self.train_dataset)}
            Validation Size:          {len(self.val_dataset)}
            Checkpoints Saved:    {opt.checkpoint_dir}
            Checkpoints Loaded:  {opt.net_checkpath}
            Device:                      {device}
        ''')

        # resume model
        if self.opt.resume:
            resume_flag = False
            if self.opt.net_checkpath:
                self.net = self.load_model(net=self.net, net_checkpath=self.opt.net_checkpath, output=True)
                resume_flag = True
            assert resume_flag
        else:  # try init param
            try:
                self.weights_init(self.net)
            except Exception as err:
                print(f'init failed: {err}')

        self.net = self.net.to(device)
        self.ema_net = self.ema_net.to(device)

        if opt.dist:
            self.net = torch.nn.parallel.DistributedDataParallel(self.net, device_ids=[opt.local_rank],
                                                                 output_device=opt.local_rank,
                                                                 find_unused_parameters=True)
            train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset)
            val_sampler = torch.utils.data.distributed.DistributedSampler(self.val_dataset)
        else:
            train_sampler = None
            val_sampler = None

        self.reset_parameters()
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=opt.batch_size,
            num_workers=opt.num_workers,
            sampler=train_sampler,
            pin_memory=True,
            shuffle=False,
            # shuffle= (train_sampler is None),
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=1,
            num_workers=opt.num_workers,
            sampler=val_sampler,
        )

        # init and resume optimizer
        self.optimizer = self.get_optimizer(self.net)
        # Not use scheduler at current stage
        self.scheduler = self.get_scheduler(self.optimizer)

        if self.opt.resume_opt:
            self.resume_opt()
            print(f'resumed optimizers at epoch {self.epoch}.')

        # start training
        start_epoch = self.epoch
        self.iter = 0
        for self.epoch in range(start_epoch, opt.epochs):
            print(f'start training epoch: {self.epoch}')
            if self.opt.dist:
                self.train_loader.sampler.set_epoch(self.epoch)
            self.train()
            if self.scheduler is not None:
                self.scheduler.step()
            save_condition = ((self.epoch + 1) % self.opt.save_epochs == 0) or ((self.epoch + 1) == self.opt.epochs)

            if self.opt.local_rank == 0 and self.epoch >= 0 and save_condition:
                self.save_model(net=self.net, net_name='colddiff', ddp_model= self.opt.dist)
                self.save_model(net=self.ema_net, net_name='colddiff_ema')
                if self.epoch >= 5:  # temporarily save less
                    self.save_opt(optimizer=self.optimizer, scheduler= self.scheduler, opt_name='colddiff')
            if self.epoch % self.opt.val_interval == 0 or (self.epoch + 1) == self.opt.epochs:
                val_save_conditon = self.val()
                if val_save_conditon:
                    self.save_model(net=self.net, net_name='colddiff', best_val_model= True, ddp_model= self.opt.dist)
                    self.save_model(net=self.ema_net, net_name='colddiff_ema')
        if self.tb_writer is not None:
            self.tb_writer.close()
