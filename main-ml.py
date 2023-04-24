import logging

import hydra
from omegaconf import DictConfig, OmegaConf

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from cadml.datamodules import ROIClassificationDataModule
from cadml.models import SimpleCNN, ResNet
from cadml.tasks import ROIClassificationTask
from cadml.callbacks import MetricsLoggingCallback


logger = logging.getLogger(__name__)


def train(cfg: DictConfig, datamodule: pl.LightningDataModule, task: pl.LightningModule):
    datamodule.setup()

    # TODO add metrics writing to the checkpoint??
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=3,
        monitor="f1",
        mode="max",
        dirpath="output/",
        filename=f"cad-{cfg.model.name}-{{epoch:03d}}-{{f1:.4f}}",
    )
    # wandb_logger = WandbLogger(project="???")
    trainer = pl.Trainer(
        accelerator=cfg.task.accelerator,
        # logger=wandb_logger,
        devices=cfg.task.devices,
        max_epochs=cfg.task.max_epochs,
        # default_root_dir='output',
        callbacks=[MetricsLoggingCallback(), checkpoint_callback],
        # fast_dev_run=True,
        # overfit_batches=10,
        # profiler='advanced',
    )
    # if no checkpoint_path is passed, then it is None, thus the model will start from the very beginning
    trainer.fit(task, datamodule=datamodule, ckpt_path=cfg.model.checkpoint_path)
    trainer.test(task, datamodule=datamodule, ckpt_path='best')


def evaluate(cfg: DictConfig, datamodule: pl.LightningDataModule, task: pl.LightningModule):
    if cfg.model.checkpoint_path is None:
        raise ValueError('no checkpoint path has been passed')

    datamodule.setup('test')

    trainer = pl.Trainer(
        accelerator=cfg.task.accelerator,
        devices=cfg.task.devices,
    )
    trainer.test(task, datamodule=datamodule, ckpt_path=cfg.model.checkpoint_path)


def inference(cfg: DictConfig, datamodule: pl.LightningDataModule, task: pl.LightningModule):
    datamodule.setup('predict')
    raise NotImplementedError()


@hydra.main(config_path='cadml/conf', config_name='config', version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg.seed)

    # FIXME refactor this?
    if cfg.datamodule.name == 'roi-classification':
        datamodule = ROIClassificationDataModule(cfg)
    else:
        raise ValueError('unknown datamodule, can be either `roi-classification` or ...')

    if cfg.model.name == 'simple-cnn':
        model = SimpleCNN(cfg)
    elif cfg.model.name == 'resnet':
        model = ResNet(cfg)
    else:
        raise ValueError('unknown model, can be either `simple-cnn` or ...')

    if cfg.task.name == 'roi-classification':
        task = ROIClassificationTask(cfg, model)
    else:
        raise ValueError('unknown task, can be either `roi-classification` or ...')

    if cfg.stage == 'train':
        train(cfg, datamodule, task)
    elif cfg.stage == 'evaluate':
        evaluate(cfg, datamodule, task)
    elif cfg.stage == 'inference':
        inference(cfg, datamodule, task)
    else:
        raise ValueError('unknown stage, can be either `train`, `evaluate` or `inference`')


if __name__ == '__main__':
    main()
