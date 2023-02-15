from nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers import BaseTokenizer
import json
import os
from pathlib import Path


class SimpleTokenizer(BaseTokenizer):
    def __init__(
            self,
            token_list_path,
            manifest_path,
    ):
        self.token_list_path = token_list_path
        if not os.path.exists(self.token_list_path):
            tokens = self._construct_vocab(manifest=manifest_path)
        else:
            tokens = [x.replace("\n", "") for x in open(token_list_path).readlines()]
        print("len(tokens) = ", len(tokens))
        super().__init__(tokens=tokens, sep=' ')

    def _construct_vocab(self, manifest):
        files = [json.loads(x) for x in open(manifest).readlines()]
        texts = [x['text'] for x in files]
        vocab = set()
        for x in texts:
            tmp = [a for a in x.split() if len(a) > 0]
            vocab.update(tmp)
        with open(self.token_list_path, 'w') as fp:
            fp.writelines([x+"\n" for x in vocab])
        return vocab

    def encode(self, text):
        cs = []
        for c in text.split():
            if c == self.sep:
                continue
            else:
                cs.append(self._token2id[c])

        # remove trailing spaces:
        if cs:
            while cs[-1] == self.sep:
                cs.pop()

        return cs


if __name__=="__main__":
    tokenizer = SimpleTokenizer(manifest_path= '../processed/manifest_tts.json', token_list_path="../processed/token_list.txt")
    sample = "n k a g E q A p h A N j A j U s e $"
    encoded = tokenizer.encode(sample)
    print(encoded)
    decoded = tokenizer.decode(encoded)
    print(decoded)
    print(sample == decoded)
