import os
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import compute_statistics_of_path
from torchvision.utils import save_image
from tqdm import tqdm
import scipy.linalg
import torch
import warnings

class FIDCallback(Callback):
    """チェックポイント保存時にFIDスコアを計算するCallback"""
    
    def __init__(self, dataset="cifar10", real_path=None, n_samples=50000, batch_size=100):
        super().__init__()
        self.dataset = dataset
        self.real_path = real_path
        self.n_samples = n_samples
        self.batch_size = batch_size
        
        # Inceptionモデルの初期化
        self.inception = None
    
    def setup(self, trainer, pl_module, stage=None):
        """実データのパスとFID計算用モデルの設定"""
        if stage == "fit":
            self.device = pl_module.device
            
            # Inceptionモデルの初期化
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
            self.inception = InceptionV3([block_idx]).eval().to(self.device)
            
            # 実データのパスを設定
            if self.dataset == "cifar10":
                self.real_path = "./data/cifar10/train"
                if not os.path.exists(self.real_path):
                    from torchvision.datasets import CIFAR10
                    dataset = CIFAR10(root="./data", train=True, download=True)
                    dataloader = torch.utils.DataLoader(dataset, batch_size=1, suffle=False)
                    os.makedirs(self.real_path, exist_ok=True)
                    for i, (x, _) in enumerate(dataloader):
                        save_image(x, os.path.join(self.real_path, f"{i:06d}.png"))
            elif not self.real_path:
                raise ValueError("For datasets other than CIFAR10, real_path must be specified")

    def calculate_fid(self, pl_module, samples_dir):
        """FIDスコアを計算"""
        # 実データの統計値を計算
        stats_path = os.path.join(pl_module.logger.experiment.log_dir, "real_stats.npy")
        if os.path.exists(stats_path):
            data = np.load(stats_path)
            m1, s1 = data['mu'], data['sigma']
        else:
            m1, s1 = compute_statistics_of_path(
                path=self.real_path,
                batch_size=self.batch_size,
                device=self.device,
                dims=2048,
                model=self.inception
            )
            np.savez(stats_path, mu=m1, sigma=s1)
        
        # 生成画像の統計値を計算
        m2, s2 = compute_statistics_of_path(
            path=samples_dir,
            batch_size=self.batch_size,
            device=self.device,
            dims=2048,
            model=self.inception
        )
        
        # FIDスコアの計算
        return self.calculate_fid_from_stats(m1, s1, m2, s2)
    
    def calculate_fid_from_stats(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """統計値からFIDスコアを計算"""
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, "Mean vectors have different lengths"
        assert sigma1.shape == sigma2.shape, "Covariances have different dimensions"

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = f"fid calculation produces singular product; adding {eps} to diagonal of cov estimates"
            warnings.warn(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError(f"Imaginary component {m}")
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """チェックポイント保存時にFIDを計算"""
        if trainer.is_global_zero:  # メインプロセスでのみ実行
            # サンプル生成用ディレクトリ
            samples_dir = "./generated_samples/{}".format("cifar10")
            os.makedirs(samples_dir, exist_ok=True)
            
            # サンプル生成
            print(f"\nGenerating {self.n_samples} samples for FID calculation...")
            with torch.no_grad():
                for i in tqdm(range(0, self.n_samples, self.batch_size)):
                    batch_size = min(self.batch_size, self.n_samples - i)
                    samples = pl_module.sample(batch_size, self.device)
                    
                    for j, sample in enumerate(samples):
                        save_image(
                            sample,
                            os.path.join(samples_dir, f'sample_{i+j}.png'),
                            normalize=True
                        )
            
            # FID計算
            print("Calculating FID score...")
            fid_score = self.calculate_fid(pl_module, samples_dir)
            
            # 結果の保存
            results_file = os.path.join(trainer.logger.log_dir, f"fid_epoch_{trainer.current_epoch}.txt")
            with open(results_file, "w") as f:
                f.write(f"Epoch: {trainer.current_epoch}\n")
                f.write(f"FID: {fid_score:.2f}\n")
            
            # TensorBoardにロギング
            trainer.logger.experiment.add_scalar('metrics/fid', fid_score, trainer.current_epoch)
            
            print(f"FID score at epoch {trainer.current_epoch}: {fid_score:.2f}")
