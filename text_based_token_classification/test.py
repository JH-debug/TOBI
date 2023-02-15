import logging
import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.loggers.wandb import WandbLogger
import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer
from data import TokenClassificationDataModule
from model import TokenClassificationTransformer


@hydra.main(version_base=None, config_path='../conf', config_name='token_classification_test')
def main(config: DictConfig):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(config)}')
    seed_everything(config.seed)

    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model)
    dm = TokenClassificationDataModule(batch_size=config.batch_size,
                                       preprocessing_num_workers=config.num_workers,
                                       data_type=config.data_type,
                                       label_all_tokens=False,
                                       revision='master',
                                       train_file=config.train_file,
                                       validation_file=config.validation_file,
                                       test_file=config.test_file,
                                       tokenizer=tokenizer)

    model = TokenClassificationTransformer(pretrained_model_name_or_path=config.pretrained_model,
                                           # tokenizer=tokenizer,
                                           labels=dm.num_classes)

    trainer = pl.Trainer(accelerator='auto',
                         devices=config.devices,
                         max_epochs=config.max_epochs,
                         accumulate_grad_batches=config.accumulate_grad_batches,
                         enable_checkpointing=True,
                         enable_progress_bar=True,
                         enable_model_summary=True,
                         logger=[WandbLogger(project=config.project, name=config.name)],
                         )
    trainer.test(model=model, ckpt_path=config.model_path, datamodule=dm)


if __name__ == "__main__":
    main()
