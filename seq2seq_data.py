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
from typing import Any

from transformers import PreTrainedTokenizerBase

from lightning_transformers.task.nlp.translation.data import TranslationDataModule


class Seq2SeqDataModule(TranslationDataModule):
    def __init__(self, *args,data_type: str = "grapheme",  **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.data_type = data_type

    @property
    def source_target_column_names(self):
        if self.data_type == "grapheme":
            src_text_column_name = "text"
            tgt_text_column_name = "text_label"
        else:
            src_text_column_name = "phoneme"
            tgt_text_column_name = "phoneme_label"
        return src_text_column_name, tgt_text_column_name

    @staticmethod
    def convert_to_features(
        examples: Any,
        tokenizer: PreTrainedTokenizerBase,
        padding: str,
        max_source_length: int,
        max_target_length: int,
        src_text_column_name: str,
        tgt_text_column_name: str,
    ):
        # inputs = [ex[src_text_column_name] for ex in examples]
        # targets = [ex[tgt_text_column_name] for ex in examples]
        inputs = examples[src_text_column_name]
        targets = examples[tgt_text_column_name]
        model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("KETI-AIR/ke-t5-base-ko")
    dm = Seq2SeqDataModule(train_file = "processed/seq2seq_all_data.json",
                           batch_size=1,
                           tokenizer=tokenizer)
    dm.setup('fit')
    print(next(iter(dm.train_dataloader())))