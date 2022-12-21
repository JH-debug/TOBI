import logging
import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.loggers.wandb import WandbLogger
import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer
from seq2seq_data import Seq2SeqDataModule
from seq2seq_model import Seq2SeqModelTransformer


@hydra.main(version_base=None, config_path='conf', config_name='seq2seq')
def main(config: DictConfig):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(config)}')
    seed_everything(config.seed)

    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model)
    dm = Seq2SeqDataModule(batch_size=config.batch_size,
                           preprocessing_num_workers=config.num_workers,
                           train_file=config.train_file,
                           validation_file=config.validation_file,
                           tokenizer=tokenizer)

    model = Seq2SeqModelTransformer(pretrained_model_name_or_path=config.pretrained_model)
    model.train()

    lr_monitor = pl.callbacks.LearningRateMonitor()
    early_stop = pl.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=5)
    checkponiter = pl.callbacks.ModelCheckpoint(dirpath=config.checkpoint_dir,
                                                filename='seq2seq_' + config.data_type + '_t5_base_{epoch:d}-{val_loss:.2f}',
                                                verbose=True, save_top_k=2, monitor='val_accuracy',
                                                mode='max', save_on_train_epoch_end=True, # save_last=True
                                                )

    trainer = pl.Trainer(accelerator=config.accelerator,
                         devices=config.devices,
                         check_val_every_n_epoch=config.check_val_every_n_epoch,
                         log_every_n_steps=config.log_every_n_steps,
                         accumulate_grad_batches=config.accumulate_grad_batches,
                         enable_checkpointing = True,
                         enable_progress_bar = True,
                         enable_model_summary = True,
                         callbacks = [lr_monitor, checkponiter, early_stop],
                         logger = WandbLogger(project=config.project, name=config.name),
                         )
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()