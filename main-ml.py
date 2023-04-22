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

   # trainer = pl.Trainer(accelerator=cfg.trainer.accelerator, devices=1, max_epochs=cfg.trainer.max_epochs)

   # lr_finder = trainer.tuner.lr_find(model)

    # Results can be found in
   # lr_finder.results

    # Plot with
#    fig = lr_finder.plot(suggest=True)
#    fig.show()

    # Pick point based on plot, or get suggestion
#    new_lr = lr_finder.suggestion()
#    print(new_lr)

    # TODO select callbacks based on config?

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=3,
        monitor="f1",
        mode="max",
        dirpath="output/",
        filename=f"cad-{cfg.model.name}-{{epoch:03d}}-{{f1:.4f}}",  # TODO better path? maybe specified in the config
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


def evaluate(cfg: DictConfig, datamodule: pl.LightningDataModule, task: pl.LightningModule):
    datamodule.setup('test')
    raise NotImplementedError()


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
