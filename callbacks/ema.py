import pytorch_lightning as pl
import torch

class EMACallback(pl.Callback):
    """Exponential Moving Average (EMA) callback for model parameters"""
    
    def __init__(self, decay=0.9999):
        """
        Args:
            decay: EMAの減衰率
        """
        super().__init__()
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
    def on_fit_start(self, trainer, pl_module):
        """訓練開始時にEMAパラメータを初期化"""
        for name, param in pl_module.netG.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """各バッチ後にEMAを更新"""
        for name, param in pl_module.netG.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.shadow[name] * self.decay + 
                    param.data * (1 - self.decay)
                )
                
    def on_validation_start(self, trainer, pl_module):
        """検証開始時にモデルのパラメータをEMAに置き換え"""
        for name, param in pl_module.netG.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
                
    def on_validation_end(self, trainer, pl_module):
        """検証終了時にモデルのパラメータを元に戻す"""
        for name, param in pl_module.netG.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup.clear()
        
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """チェックポイント保存時にEMAの状態も保存"""
        return {"shadow": self.shadow}
        
    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        """チェックポイント読み込み時にEMAの状態を復元"""
        self.shadow = checkpoint["shadow"]
