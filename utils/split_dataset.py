import json
from sklearn.model_selection import train_test_split


def make_split_data(type="token_classfication"):
    if type == "token_classification":
        with open("../processed/all_data.json", "r") as f:
            data = json.load(f)

        train, test = train_test_split(data, test_size=0.2, random_state=42)
        val, test = train_test_split(test, test_size=0.5)

        print(len(train), len(val), len(test))

        for split in ('train', 'val', 'test'):
            if split == 'train':
                split_data = train
            elif split == 'val':
                split_data = val
            else:
                split_data = test
            with open(f"processed/{split}.json", "w", encoding='UTF-8') as f:
                json.dump(split_data, f, indent=1, ensure_ascii=False)

    else:
        with open("../processed/seq2seq_all_data.json", "r") as f:
            data = json.load(f)

        train, test = train_test_split(data, test_size=0.2, random_state=42)
        val, test = train_test_split(test, test_size=0.5)

        print(len(train), len(val), len(test))

        for split in ('train', 'val', 'test'):
            if split == 'train':
                split_data = train
            elif split == 'val':
                split_data = val
            else:
                split_data = test
            with open(f"../processed/seq2seq_{split}.json", "w", encoding='UTF-8') as f:
                json.dump(split_data, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    make_split_data(type="seq2seq")