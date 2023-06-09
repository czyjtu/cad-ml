import logging

import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from cadml.datamodules import ROIClassificationDataModule
from cadml.models import SimpleCNN, ResNet
from cadml.tasks import ROIClassificationTask
from cadml.callbacks import MetricsLoggingCallback
import torch

logger = logging.getLogger(__name__)


def train(
    cfg: DictConfig, datamodule: pl.LightningDataModule, task: pl.LightningModule
):
    datamodule.setup()

    # TODO add metrics writing to the checkpoint??
    wandb_logger = WandbLogger(project="cad-ml")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=3,
        monitor="f1",
        mode="max",
        dirpath="output/",
        filename=f"cad-{cfg.model.name}-{wandb_logger.experiment.name}-{{epoch:03d}}-{{f1:.4f}}",
    )
    trainer = pl.Trainer(
        accelerator=cfg.task.accelerator,
        logger=wandb_logger,
        devices=cfg.task.devices,
        max_epochs=cfg.task.max_epochs,
        # default_root_dir='output',
        callbacks=[MetricsLoggingCallback(), checkpoint_callback],
        # fast_dev_run=True,
        # overfit_batches=10,
        # profiler='simple',
    )
    # if no checkpoint_path is passed, then it is None, thus the model will start from the very beginning
    trainer.fit(task, datamodule=datamodule, ckpt_path=cfg.model.checkpoint_path)
    trainer.test(task, datamodule=datamodule, ckpt_path="best")


def evaluate(
    cfg: DictConfig, datamodule: pl.LightningDataModule, task: pl.LightningModule
):
    if cfg.model.checkpoint_path is None:
        raise ValueError("no checkpoint path has been passed")

    datamodule.setup("test")

    trainer = pl.Trainer(
        accelerator=cfg.task.accelerator,
        devices=cfg.task.devices,
        # precision=32
    )
    trainer.test(task, datamodule=datamodule, ckpt_path=cfg.model.checkpoint_path)


def inference(
    cfg: DictConfig, datamodule: pl.LightningDataModule, task: pl.LightningModule
):
    datamodule.setup("predict")
    raise NotImplementedError()


def explore(
    cfg: DictConfig, datamodule: pl.LightningDataModule, task: pl.LightningModule
):
    datamodule.setup()
    for idx in range(len(datamodule.train_dataset)):
        image, label = datamodule.train_dataset[idx]
        print(image)
        plt.title(f'Label: {label}')
        plt.imshow(image.squeeze(0), cmap="gray")
        plt.show()

    # from coronaryx import plot_scan, plot_branch
    # for item in datamodule.train_dataset.original_dataset:
    #     plot_branch(item)


@hydra.main(config_path="cadml/conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg.seed)

    # FIXME refactor this?
    if cfg.datamodule.name == "roi-classification":
        datamodule = ROIClassificationDataModule(cfg)
    else:
        raise ValueError(
            "unknown datamodule, can be either `roi-classification` or ..."
        )

    if cfg.model.name == "simple-cnn":
        model = SimpleCNN(cfg)
    elif cfg.model.name == "resnet":
        model = ResNet(cfg)
    else:
        raise ValueError("unknown model, can be either `simple-cnn` or ...")

    if cfg.task.name == "roi-classification":
        task = ROIClassificationTask(cfg, model)
    else:
        raise ValueError("unknown task, can be either `roi-classification` or ...")

    if cfg.stage == "train":
        train(cfg, datamodule, task)
    elif cfg.stage == "evaluate":
        evaluate(cfg, datamodule, task)
    elif cfg.stage == "inference":
        inference(cfg, datamodule, task)
    elif cfg.stage == 'explore':
        explore(cfg, datamodule, task)
    else:
        raise ValueError(
            "unknown stage, can be either `train`, `evaluate` or `inference`"
        )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    main()
