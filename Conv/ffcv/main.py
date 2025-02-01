import torch
from model import Net
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger, NeptuneLogger
from lightning import seed_everything
from data import DataModule
from config_manager import get_config
from lightning.pytorch.callbacks import LearningRateMonitor, TQDMProgressBar, ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy

import os 
from dotenv import load_dotenv
from loguru import logger

import logging

import warnings
warnings.filterwarnings('ignore', message='Grad strides do not match bucket view strides')

#SECTION: This is for fix warning from neptune, see https://github.com/neptune-ai/neptune-client/issues/1702
class _FilterCallback(logging.Filterer):
    def filter(self, record: logging.LogRecord):
        return not (
            record.name == "neptune"
            and record.getMessage().startswith(
                "Error occurred during asynchronous operation processing: X-coordinates (step) must be strictly increasing for series attribute"
            )
        )

logging.getLogger("neptune").addFilter(_FilterCallback())
#ENDSECTION

def main():
    # Others 
    torch.set_float32_matmul_precision('high')  # For A100 to trade off between precision and performance
    load_dotenv()
    
    # Load configuration
    config = get_config()

    # Keep the same seed for reproducibility
    seed_everything(0, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # loggers for experiments
    tb_logger = TensorBoardLogger(name=config['experiment']['name'], save_dir="logs")

    neptune_logger = NeptuneLogger(
        api_key=os.getenv('NEPTUNE_API_TOKEN'),  # replace with your own
        project="dylanheddeldy/MaxSup",  # format "workspace-name/project-name"
        tags=["test"],  # optional
    )

    # callbacks
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'/datadrive2/hengl/icml2025/{config["model"]["model"]}',
        filename=f"{config["experiment"]["name"]}" + "-{epoch:02d}-{Val_Acc_Top1:.2f}",
        save_top_k=1,
        mode='max',
        save_last=True,
        monitor="Val_Acc_Top1",
    )
    progress_bar = TQDMProgressBar(leave=True)
    callbacks = [lr_monitor, checkpoint_callback, progress_bar]

    
    # data module
    data_module = DataModule(config['data'])

    trainer = L.Trainer(
        accelerator="gpu",
        devices=config['train']['world_size'],
        max_epochs=config['train']['epochs'],
        logger=[tb_logger, neptune_logger],
        callbacks=callbacks,
        precision="bf16-mixed",
        fast_dev_run=config['experiment']['fast_run'],
        strategy=DDPStrategy(find_unused_parameters=False),
        sync_batchnorm=True,
    )

    model = Net(config['model'])
    trainer.fit(model, datamodule=data_module)



if __name__ == "__main__":
    main()