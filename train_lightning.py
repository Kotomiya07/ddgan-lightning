#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DDGAN (Denoising Diffusion GAN) Training Script
PyTorch Lightning implementation for training DDGAN models.
"""

import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from models.ddgan import DDGAN
from data.datasets import DDGANDataModule
from callbacks.ema import EMACallback

def main(args):
    # シード設定
    pl.seed_everything(args.seed)

    # モデルの初期化
    model = DDGAN(args)

    # データモジュールの初期化
    datamodule = DDGANDataModule(args)

    # コールバックの設定
    callbacks = [
        ModelCheckpoint(
            dirpath=f"./saved_info/dd_gan/{args.dataset}/{args.exp}",
            filename="{epoch}",
            save_top_k=2,
            monitor="train/g_loss",
            mode="min",
            save_last=True,
            every_n_epochs=args.save_ckpt_every
        ),
        LearningRateMonitor(logging_interval="step")
    ]

    if args.use_ema:
        callbacks.append(EMACallback(decay=args.ema_decay))

    # ロガーの設定
    logger = TensorBoardLogger(
        save_dir=f"./saved_info/dd_gan/{args.dataset}",
        name=args.exp,
        default_hp_metric=False
    )

    # トレーナーの設定
    # チェックポイントパスの設定
    ckpt_path = None
    if args.resume:
        ckpt_dir = f"./saved_info/dd_gan/{args.dataset}/{args.exp}"
        if os.path.exists(os.path.join(ckpt_dir, "last.ckpt")):
            ckpt_path = os.path.join(ckpt_dir, "last.ckpt")
        else:
            print("チェックポイントが見つかりません。最初から学習を開始します。")

    # 学習戦略の設定
    strategy = "auto"
    if args.num_process_per_node > 1:
        strategy = DDPStrategy(
            find_unused_parameters=False,
            process_group_backend="nccl"
        )

    trainer = pl.Trainer(
        max_epochs=args.num_epoch,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.num_process_per_node,
        num_nodes=args.num_proc_node,
        strategy=strategy,
        callbacks=callbacks,
        logger=logger,
        precision=32,
        # deterministic=Trueの場合はbenchmark=Falseにする
        benchmark=False if args.seed is not None else True,
        deterministic=True if args.seed is not None else False,
        log_every_n_steps=100,
    )

    # トレーニングの実行
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser('DDGAN training')
    
    # 基本的な設定
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--exp', required=True, help='experiment name')
    parser.add_argument('--dataset', default='cifar10', help='dataset name')
    parser.add_argument('--resume', action='store_true', default=False)

    # モデルの設定
    parser.add_argument('--image_size', type=int, default=32, help='size of image')
    parser.add_argument('--num_channels', type=int, default=3, help='channel of image')
    parser.add_argument('--centered', action='store_false', default=True, help='-1,1 scale')
    parser.add_argument('--use_geometric', action='store_true', default=False)
    parser.add_argument('--beta_min', type=float, default=0.1, help='beta_min for diffusion')
    parser.add_argument('--beta_max', type=float, default=20., help='beta_max for diffusion')

    # ネットワークアーキテクチャ
    parser.add_argument('--num_channels_dae', type=int, default=128, help='number of initial channels in denoising model')
    parser.add_argument('--n_mlp', type=int, default=3, help='number of mlp layers for z')
    parser.add_argument('--ch_mult', nargs='+', type=int)
    parser.add_argument('--num_res_blocks', type=int, default=2, help='number of resnet blocks per scale')
    parser.add_argument('--attn_resolutions', default=(16,), help='resolution of applying attention')
    parser.add_argument('--dropout', type=float, default=0., help='drop-out rate')
    parser.add_argument('--resamp_with_conv', action='store_false', default=True, help='always up/down sampling with conv')
    parser.add_argument('--conditional', action='store_false', default=True, help='noise conditional')
    parser.add_argument('--fir', action='store_false', default=True, help='FIR')
    parser.add_argument('--fir_kernel', default=[1, 3, 3, 1], help='FIR kernel')
    parser.add_argument('--skip_rescale', action='store_false', default=True, help='skip rescale')
    parser.add_argument('--resblock_type', default='biggan', help='tyle of resnet block')
    parser.add_argument('--progressive', type=str, default='none', choices=['none', 'output_skip', 'residual'])
    parser.add_argument('--progressive_input', type=str, default='residual', choices=['none', 'input_skip', 'residual'])
    parser.add_argument('--progressive_combine', type=str, default='sum', choices=['sum', 'cat'])
    parser.add_argument('--embedding_type', type=str, default='positional', choices=['positional', 'fourier'])
    parser.add_argument('--fourier_scale', type=float, default=16., help='scale of fourier transform')
    parser.add_argument('--not_use_tanh', action='store_true', default=False)

    # トレーニングの設定
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--num_timesteps', type=int, default=4)
    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
    parser.add_argument('--num_epoch', type=int, default=1200)
    parser.add_argument('--ngf', type=int, default=64)

    # オプティマイザの設定
    parser.add_argument('--lr_g', type=float, default=1.5e-4, help='learning rate g')
    parser.add_argument('--lr_d', type=float, default=1e-4, help='learning rate d')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.9, help='beta2 for adam')
    parser.add_argument('--no_lr_decay', action='store_true', default=False)

    # EMAの設定
    parser.add_argument('--use_ema', action='store_true', default=False, help='use EMA or not')
    parser.add_argument('--ema_decay', type=float, default=0.9999, help='decay rate for EMA')

    # 正則化の設定
    parser.add_argument('--r1_gamma', type=float, default=0.05, help='coef for r1 reg')
    parser.add_argument('--lazy_reg', type=int, default=None, help='lazy regularization.')

    # チェックポイントの設定
    parser.add_argument('--save_content_every', type=int, default=5, help='save content for resuming every x epochs')
    parser.add_argument('--save_ckpt_every', type=int, default=5, help='save ckpt every x epochs')

    # 分散学習の設定
    parser.add_argument('--num_proc_node', type=int, default=1, help='The number of nodes in multi node env.')
    parser.add_argument('--num_process_per_node', type=int, default=1, help='number of gpus')
    parser.add_argument('--node_rank', type=int, default=0, help='The index of node.')
    parser.add_argument('--master_address', type=str, default='127.0.0.1', help='address for master')

    args = parser.parse_args()
    args.world_size = args.num_proc_node * args.num_process_per_node

    main(args)
