# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Dict, List, Type, Union
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

import torch
import transformers
from torchmetrics import Accuracy, F1Score, Precision, Recall, ConfusionMatrix
from transformers.models.auto.auto_factory import _BaseAutoModelClass

from lightning_transformers.core import TaskTransformer


class TokenClassificationTransformer(TaskTransformer):
    """Defines ``LightningModule`` for the Text Classification Task.

    Args:
        *args: :class:`lightning_transformers.core.model.TaskTransformer` arguments.
        downstream_model_type: Downstream HuggingFace AutoModel to load.
            (default ``transformers.AutoModelForTokenClassification``)
        **kwargs: :class:`lightning_transformers.core.model.TaskTransformer` arguments.
    """

    def __init__(
        self,
        *args,
        labels: Union[int, List[str]],
        downstream_model_type: Type[_BaseAutoModelClass] = transformers.AutoModelForTokenClassification,
        **kwargs,
    ) -> None:
        num_labels = labels if isinstance(labels, int) else len(labels)
        super().__init__(downstream_model_type, *args, num_labels=num_labels, **kwargs)
        self._num_labels = num_labels
        self.metrics = {}

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        outputs = self.model(**batch)
        loss = outputs[0]
        self.log("train_loss", loss)
        return loss

    def common_step(self, prefix: str, batch: Any) -> torch.Tensor:
        outputs = self.model(**batch)
        loss, logits = outputs[:2]
        preds = torch.argmax(logits, dim=2)
        metric_dict = self.compute_metrics(preds, batch["labels"], mode=prefix)
        self.log_dict(metric_dict, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{prefix}_loss", loss, prog_bar=True, sync_dist=True)
        if prefix == "test":
            labels = batch["labels"]
            preds = preds[labels != -100]
            labels = labels[labels != -100]
            return {'loss': loss, 'preds': preds, 'labels': labels}
        return loss

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        return self.common_step("val", batch)

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        return self.common_step("test", batch)

    def test_epoch_end(self, outputs: List[Any]) -> None:
        preds = torch.cat([x['preds'] for x in outputs])
        labels = torch.cat([x['labels'] for x in outputs])
        confusion_matrix = self.confusion_matrix(preds, labels).detach().cpu().numpy().astype(int)
        print(confusion_matrix)

        df_cm = pd.DataFrame(confusion_matrix, index = range(2), columns=range(2))
        print(df_cm)
        plt.figure(figsize = (10,7))
        fig_ = sns.heatmap(df_cm, annot=True, cmap='Spectral', fmt='d').get_figure()
        plt.close(fig_)

        wandb_logger = self.logger.experiment
        wandb_logger.log({"confusion_matrix": wandb.Image(fig_)})

    def configure_metrics(self, _) -> None:
        self.prec = Precision(task='binary', num_classes=self.num_labels)
        self.recall = Recall(task='binary', num_classes=self.num_labels)
        self.f1 = F1Score(task='binary', num_classes=self.num_labels)
        self.acc = Accuracy(task='binary')
        self.metrics = {"precision": self.prec, "recall": self.recall, "accuracy": self.acc, "f1": self.f1}
        self.confusion_matrix = ConfusionMatrix(task='binary', num_classes=self.num_labels)

    @property
    def num_labels(self) -> int:
        return self._num_labels

    def compute_metrics(self, predictions, labels, mode="val") -> Dict[str, torch.Tensor]:
        # Remove ignored index (special tokens)
        predictions = predictions[labels != -100]
        labels = labels[labels != -100]
        # Not required by all models. Only required for classification
        return {f"{mode}_{k}": metric(predictions, labels) for k, metric in self.metrics.items()}

    @property
    def hf_pipeline_task(self) -> str:
        return "ner"