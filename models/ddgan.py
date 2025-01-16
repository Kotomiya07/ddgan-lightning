import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Adam
import torchvision
from score_sde.models.discriminator import Discriminator_large, Discriminator_small
from score_sde.models.ncsnpp_generator_adagn import NCSNpp
from utils.diffusion import DiffusionCoefficients, PosteriorCoefficients, q_sample_pairs, sample_posterior

class DDGAN(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.automatic_optimization = False  # 手動最適化モードを有効化
        
        # Generator
        self.netG = NCSNpp(args)
        
        # Discriminator
        if args.dataset in ['cifar10', 'stackmnist']:    
            self.netD = Discriminator_small(
                nc=2*args.num_channels,
                ngf=args.ngf,
                t_emb_dim=args.t_emb_dim,
                act=nn.LeakyReLU(0.2)
            )
        else:
            self.netD = Discriminator_large(
                nc=2*args.num_channels,
                ngf=args.ngf,
                t_emb_dim=args.t_emb_dim,
                act=nn.LeakyReLU(0.2)
            )
        
        # Time scheduleの登録（デバイスに依存しない部分）
        self.register_buffer('T', self.get_time_schedule())

    def setup(self, stage=None):
        """モデルのセットアップ時にデバイスが確定した後で拡散係数を初期化"""
        super().setup(stage)
        self.coeff = self.init_diffusion_coefficients()
        self.pos_coeff = self.init_posterior_coefficients()

    def get_time_schedule(self):
        n_timestep = self.args.num_timesteps
        eps_small = 1e-3
        t = torch.arange(0, n_timestep + 1, dtype=torch.float64)
        t = t / n_timestep
        t = t * (1. - eps_small) + eps_small
        return t

    def init_diffusion_coefficients(self):
        return DiffusionCoefficients(self.args, self.device)

    def init_posterior_coefficients(self):
        return PosteriorCoefficients(self.args, self.device)

    def configure_optimizers(self):
        opt_g = Adam(self.netG.parameters(), lr=self.args.lr_g, betas=(self.args.beta1, self.args.beta2))
        opt_d = Adam(self.netD.parameters(), lr=self.args.lr_d, betas=(self.args.beta1, self.args.beta2))
        
        if not self.args.no_lr_decay:
            scheduler_g = optim.lr_scheduler.CosineAnnealingLR(opt_g, self.args.num_epoch, eta_min=1e-5)
            scheduler_d = optim.lr_scheduler.CosineAnnealingLR(opt_d, self.args.num_epoch, eta_min=1e-5)
            
            return [opt_g, opt_d], [scheduler_g, scheduler_d]
        return [opt_g, opt_d]

    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()
        
        real_data, _ = batch
        batch_size = real_data.size(0)
        t = torch.randint(0, self.args.num_timesteps, (batch_size,), device=self.device)
        
        # Train Discriminator
        opt_d.zero_grad()
        x_t, x_tp1 = self.q_sample_pairs(self.coeff, real_data, t)
        x_t.requires_grad = True
        
        # Real data
        D_real = self.netD(x_t, t, x_tp1.detach()).view(-1)
        errD_real = F.softplus(-D_real).mean()
        
        # R1 regularization
        if self.args.lazy_reg is None or self.global_step % self.args.lazy_reg == 0:
            grad_real = torch.autograd.grad(
                outputs=D_real.sum(),
                inputs=x_t,
                create_graph=True
            )[0]
            grad_penalty = (
                grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
            ).mean()
            grad_penalty = self.args.r1_gamma / 2 * grad_penalty
        else:
            grad_penalty = 0

        # Fake data
        latent_z = torch.randn(batch_size, self.args.nz, device=self.device)
        x_0_predict = self.netG(x_tp1.detach(), t, latent_z)
        x_pos_sample = self.sample_posterior(self.pos_coeff, x_0_predict, x_tp1, t)
        
        output = self.netD(x_pos_sample.detach(), t, x_tp1.detach()).view(-1)
        errD_fake = F.softplus(output).mean()
        
        errD = errD_real + errD_fake + grad_penalty
        self.manual_backward(errD)
        opt_d.step()
        
        self.log('train/d_loss', errD, on_step=True, on_epoch=True)

        # Train Generator
        opt_g.zero_grad()
        latent_z = torch.randn(batch_size, self.args.nz, device=self.device)
        x_t, x_tp1 = self.q_sample_pairs(self.coeff, real_data, t)
        
        x_0_predict = self.netG(x_tp1.detach(), t, latent_z)
        x_pos_sample = self.sample_posterior(self.pos_coeff, x_0_predict, x_tp1, t)
        
        output = self.netD(x_pos_sample, t, x_tp1.detach()).view(-1)
        errG = F.softplus(-output).mean()
        
        self.manual_backward(errG)
        opt_g.step()
        
        self.log('train/g_loss', errG, on_step=True, on_epoch=True)

    def q_sample_pairs(self, coeff, x_start, t):
        return q_sample_pairs(coeff, x_start, t)

    def sample_posterior(self, coefficients, x_0, x_t, t):
        return sample_posterior(coefficients, x_0, x_t, t)

    def sample(self, n_samples, device):
        """モデルからサンプリングを行う"""
        with torch.no_grad():
            x_t_1 = torch.randn(n_samples, self.args.num_channels, 
                              self.args.image_size, self.args.image_size, 
                              device=device)
            
            x = x_t_1
            for i in reversed(range(self.args.num_timesteps)):
                t = torch.full((n_samples,), i, dtype=torch.int64, device=device)
                latent_z = torch.randn(n_samples, self.args.nz, device=device)
                x_0 = self.netG(x, t, latent_z)
                x_new = self.sample_posterior(self.pos_coeff, x_0, x, t)
                x = x_new.detach()
                
        return x
    
    def on_train_epoch_end(self):
        """エポック終了時にサンプル画像を生成"""
        if self.global_rank == 0 and self.current_epoch % 10 == 0:
            samples = self.sample(16, self.device)
            grid = torchvision.utils.make_grid(samples, normalize=True)
            self.logger.experiment.add_image(
                'generated_images', 
                grid, 
                self.current_epoch
            )
