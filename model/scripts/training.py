import os
import torch as th

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from data.lightning_flip import FlipDataModule
from model.lightning.flip import FlipModule
from utils.configuration import Configuration
from utils.io import PeriodicCheckpoint

def train(cfg: Configuration, checkpoint_path, load_model_only = False):
    
    os.makedirs(f"out/{cfg.model_path}", exist_ok=True)
    os.makedirs(f"out/{cfg.model_path}/profiler", exist_ok=True)


    data_module = FlipDataModule(cfg)
    model       = FlipModule(cfg)

    # Load the model from the checkpoint if provided, otherwise create a new model
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        if load_model_only:
            checkpoint = th.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(checkpoint["state_dict"], strict=False)
        else:
            model = FlipModule.load_from_checkpoint(checkpoint_path, cfg=cfg, strict=False)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=f"out/{cfg.model_path}",
        filename="LociPretrainer-{epoch:02d}-{val_mean_iou:.2f}",
        save_top_k=3,
        mode="min",
        verbose=True,
    )

    periodic_min_checkpoint_callback = ModelCheckpoint(
        monitor="loss",
        dirpath=f"out/{cfg.model_path}",
        filename="LociPretrainer-{step:02d}-{mean_iou:.2f}",
        save_top_k=3,
        mode="min",
        every_n_train_steps=1000,
        verbose=True,
    )

    periodic_checkpoint_callback = PeriodicCheckpoint(
        save_path=f"out/{cfg.model_path}",
        save_every_n_steps=1000,  # Save checkpoint every 3000 global steps
    )

    #callback_list = [checkpoint_callback, periodic_checkpoint_callback, periodic_min_checkpoint_callback]
    callback_list = [periodic_checkpoint_callback]

    if th.cuda.is_available():
        trainer = pl.Trainer(
            devices=([cfg.device] if cfg.single_gpu else "auto"), 
            accumulate_grad_batches=cfg.model.gradient_accumulation_steps,
            accelerator="cuda",
            strategy="ddp_find_unused_parameters_true",
            max_epochs=cfg.epochs,
            callbacks=callback_list,
            precision=16 if cfg.model.mixed_precision else 32,
            enable_progress_bar=False,
            logger=False,
            val_check_interval=cfg.eval_interval
        )
    elif th.backends.mps.is_available():
        trainer = pl.Trainer(
            accumulate_grad_batches=cfg.model.gradient_accumulation_steps,
            accelerator="mps",
            max_epochs=cfg.epochs,
            callbacks=callback_list,
            precision=16 if cfg.model.mixed_precision else 32,
            enable_progress_bar=False,
            logger=False,
            val_check_interval=cfg.eval_interval
        )
    else:
        trainer = pl.Trainer(
            accumulate_grad_batches=cfg.model.gradient_accumulation_steps,
            accelerator="cpu",
            max_epochs=cfg.epochs,
            callbacks=callback_list,
            precision=16 if cfg.model.mixed_precision else 32,
            enable_progress_bar=False,
            logger=False,
            val_check_interval=cfg.eval_interval
        )

    if cfg.validate:
        trainer.validate(model, data_module)
    else:
        trainer.fit(model, data_module)
