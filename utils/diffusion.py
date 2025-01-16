import numpy as np
import torch

def var_func_vp(t, beta_min, beta_max):
    """Variance scheduling using variance preserving formulation"""
    log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
    var = 1. - torch.exp(2. * log_mean_coeff)
    return var

def var_func_geometric(t, beta_min, beta_max):
    """Variance scheduling using geometric formulation"""
    return beta_min * ((beta_max / beta_min) ** t)

def extract(data_input, t, shape):
    """Extract the appropriate t index for a batch of indices."""
    out = torch.gather(data_input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)
    return out

class DiffusionCoefficients:
    """拡散過程の係数を管理するクラス"""
    def __init__(self, args, device):
        """
        Args:
            args: 設定パラメータ
            device: 計算に使用するデバイス
        """
        self.device = device
        self.num_timesteps = args.num_timesteps
        self.beta_min = args.beta_min
        self.beta_max = args.beta_max
        self.use_geometric = args.use_geometric
        
        # σとαのスケジュールを計算
        self.sigmas, self.a_s, _ = self._get_sigma_schedule()
        
        # 累積積を計算
        self.a_s_cum = torch.cumprod(self.a_s, dim=0)
        self.sigmas_cum = torch.sqrt(1 - self.a_s_cum ** 2)
        
        # 前のステップのαを保存
        self.a_s_prev = self.a_s.clone()
        self.a_s_prev[-1] = 1

    def _get_sigma_schedule(self):
        """σスケジュールを計算"""
        eps_small = 1e-3
        
        t = torch.linspace(0, 1, self.num_timesteps + 1, device=self.device)
        t = t * (1. - eps_small) + eps_small
        
        if self.use_geometric:
            var = var_func_geometric(t, self.beta_min, self.beta_max)
        else:
            var = var_func_vp(t, self.beta_min, self.beta_max)
        
        alpha_bars = 1.0 - var
        betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
        
        first = torch.tensor(1e-8, device=self.device)
        betas = torch.cat((first[None], betas))
        sigmas = betas**0.5
        a_s = torch.sqrt(1-betas)
        
        return sigmas, a_s, betas

class PosteriorCoefficients:
    """事後分布のサンプリングに必要な係数を管理するクラス"""
    def __init__(self, args, device):
        """
        Args:
            args: 設定パラメータ
            device: 計算に使用するデバイス
        """
        _, _, self.betas = DiffusionCoefficients(args, device)._get_sigma_schedule()
        
        # 最初のβを除外
        self.betas = self.betas[1:]
        
        # 各種係数を計算
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat(
            (torch.tensor([1.], device=device), self.alphas_cumprod[:-1]), 0
        )
        
        # 事後分布の分散を計算
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        
        # 平均計算用の係数を準備
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.rsqrt(self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1)
        
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_cumprod)
        
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))

def q_sample(coeff, x_start, t, noise=None):
    """
    拡散過程でのサンプリング
    
    Args:
        coeff: DiffusionCoefficients インスタンス
        x_start: 開始画像
        t: タイムステップ
        noise: ノイズ（Noneの場合は生成される）
    
    Returns:
        x_t: 時刻tでのサンプル
    """
    if noise is None:
        noise = torch.randn_like(x_start)
    
    x_t = extract(coeff.a_s_cum, t, x_start.shape) * x_start + \
          extract(coeff.sigmas_cum, t, x_start.shape) * noise
    
    return x_t

def q_sample_pairs(coeff, x_start, t):
    """
    連続する時刻でのサンプルペアを生成
    
    Args:
        coeff: DiffusionCoefficients インスタンス
        x_start: 開始画像
        t: タイムステップ
    
    Returns:
        (x_t, x_t+1): サンプルペア
    """
    noise = torch.randn_like(x_start)
    x_t = q_sample(coeff, x_start, t, noise)
    x_t_plus_one = extract(coeff.a_s, t+1, x_start.shape) * x_t + \
                   extract(coeff.sigmas, t+1, x_start.shape) * noise
    
    return x_t, x_t_plus_one

def sample_posterior(coefficients, x_0, x_t, t):
    """
    事後分布からのサンプリング
    
    Args:
        coefficients: PosteriorCoefficients インスタンス
        x_0: 予測された元画像
        x_t: 現在の状態
        t: タイムステップ
    
    Returns:
        サンプリングされた次の状態
    """
    def q_posterior(x_0, x_t, t):
        mean = (
            extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0
            + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(coefficients.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(coefficients.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var_clipped
    
    def p_sample(x_0, x_t, t):
        mean, _, log_var = q_posterior(x_0, x_t, t)
        noise = torch.randn_like(x_t)
        nonzero_mask = (1 - (t == 0).float())[:, None, None, None]
        return mean + nonzero_mask * torch.exp(0.5 * log_var) * noise
            
    return p_sample(x_0, x_t, t)
