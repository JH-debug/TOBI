import logging
import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.loggers.wandb import WandbLogger
import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer
from language_modeling_data import LanguageModelingDataModule
from language_modeling_model import LanguageModelingTransformer


@hydra.main(version_base=None, config_path='../conf', config_name='language_modeling')
def main(config: DictConfig):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(config)}')
    seed_everything(config.seed)

    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model)
    dm = LanguageModelingDataModule(batch_size=config.batch_size,
                                    preprocessing_num_workers=config.num_workers,
                                    train_file=config.train_file,
                                    validation_file=config.validation_file,
                                    data_type=config.data_type,
                                    tokenizer=tokenizer,
                                    )
    model = LanguageModelingTransformer(pretrained_model_name_or_path=config.pretrained_model, tokenizer=tokenizer)
    model.train()

    lr_monitor = pl.callbacks.LearningRateMonitor()
    early_stop = pl.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=2)
    checkponiter = pl.callbacks.ModelCheckpoint(dirpath=config.checkpoint_dir,
                                                filename='LM_' + config.data_type + '_kogpt_{epoch:d}-{val_bleu_score:.2f}',
                                                verbose=True, save_top_k=2, monitor='val_bleu_score',
                                                mode='max', save_on_train_epoch_end=True
                                                )

    trainer = pl.Trainer(accelerator=config.accelerator,
                         devices=config.devices,
                         check_val_every_n_epoch=config.check_val_every_n_epoch,
                         log_every_n_steps=config.log_every_n_steps,
                         accumulate_grad_batches=config.accumulate_grad_batches,
                         enable_checkpointing = True,
                         enable_progress_bar = True,
                         enable_model_summary = True,
                         callbacks = [lr_monitor, early_stop, checkponiter],
                         logger = WandbLogger(project=config.project, name=config.name),
                         )
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()