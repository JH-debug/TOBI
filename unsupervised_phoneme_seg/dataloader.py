import os
import math
import random
import numpy as np
from tqdm import tqdm
import librosa
import torchaudio
import soundfile as sf
from boltons.fileutils import iter_find_files
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split


def collate_fn_padd(batch):
    """collate_fn_padd
    Padds batch of variable length
    :param batch:
    """
    # get sequence lengths
    spects = [t[0] for t in batch]
    segs = [t[1] for t in batch]
    labels = [t[2] for t in batch]
    lengths = [t[3] for t in batch]
    wav_path = [t[4] for t in batch]

    # pad and stack
    padded_spects = torch.nn.utils.rnn.pad_sequence(spects, batch_first=True)
    lengths = torch.LongTensor(lengths)

    return padded_spects, segs, labels, lengths, wav_path


def spectral_size(wav_len):
    layers = [(10,5,0), (8,4,0), (4,2,0), (4,2,0), (4,2,0)]
    for kernel, stride, padding in layers:
        wav_len = math.floor((wav_len + 2*padding - 1*(kernel-1) - 1)/stride + 1)
    return wav_len


def get_subset(dataset, percent):
    A_split = int(len(dataset) * percent)
    B_split = len(dataset) - A_split
    dataset, _ = torch.utils.data.random_split(dataset, [A_split, B_split])
    return dataset


def phoneme_lebels_to_frame_labels(segmentation, phonemes):
    """
    replicates phonemes to frame-wise labels
    example:
        segmentation - [0, 3, 4]
        phonemes - [a, b]
        returns - [a, a, a, b]
    :param segmentation:
    :param phonemes:
    """
    segmentation = torch.LongTensor(segmentation)
    return torch.cat([torch.LongTensor([l]).repeat(t) for (l, t) in zip(phonemes, segmentation[1:] - segmentation[:-1])])


def segmentation_to_binary_mask(segmentation):
    """
    replicates boundaries to frame-wise labels
    example:
        segmentation - [0, 3, 5]
        returns - [1, 0, 0, 1, 0, 1]
    :param segmentation:
    :param phonemes:
    """
    mask = torch.zeros(segmentation[-1] + 1).long()
    for boundary in segmentation[1:-1]:
        mask[boundary] = 1
    return mask


class phoneme_dict():
    def __init__(self, token_list_path = "../processed/token_list.txt"):
        self.token_list = [x.replace("\n", "") for x in open(token_list_path).readlines()]
        self.n_tokens = len(self.token_list)
        self.pho_to_idx = {token: i for i, token in enumerate(self.token_list)}
        self.idx_to_pho = {v:k for k,v in self.pho_to_idx.items()}

    def encode(self, phonemes):
        return [self.pho_to_idx[phoneme] for phoneme in phonemes]

    def decode(self, phoneme_idx):
        return [self.idx_to_pho[idx] for idx in phoneme_idx]


class phoneme_seg_dataset(Dataset):
    def __init__(self, config):
        super(phoneme_seg_dataset, self).__init__()
        self.config = config
        self.phn_path = config.phn_path
        self.wav_path = config.wav_path
        # self.vocab = phoneme_dict()
        self.data = list(iter_find_files(self.wav_path, "*.wav"))
        if self.config.use_only_break_index:
            self.data = self._filter_dataset()

    def process_file(self, wav_file):
        # load audio
        audio, sr = torchaudio.load(wav_file)
        audio = audio[0]
        audio_len = audio.shape[0]
        spectral_len = spectral_size(audio_len)
        len_ratio = (audio_len / spectral_len)

        # load phoneme and segmentation
        phn_file = wav_file.replace(self.wav_path, self.phn_path).replace(".wav", ".phone")
        with open(phn_file, 'r') as f:  # segmentation and phonemes
            lines = f.readlines()
            # phn = list(map(lambda x: x.replace('\n', '').split('\t'), phn))
            # phn = [x for x in phn if x[1] == ';']    # only with break index
            lines = list(map(lambda line: line.split(" "), lines))
            if self.config.use_only_break_index:
                lines = [x for x in lines if x[2].strip() == ';']  # only with break index

            # get segment times
            if self.config.use_only_break_index:
                times = torch.FloatTensor(list(map(lambda line: int(int(line[1]) / len_ratio), lines)))
            else:
                times = torch.FloatTensor(list(map(lambda line: int(int(line[1]) / len_ratio), lines)))[:-1]  # don't count end time as boundary

            # get phonemes in each segment (for k times there should be k+1 segments)
            # phn = list(map(lambda phn: phn[1].strip(), phn))
            # phn = self.vocab.encode(phn)  # phoneme encoding
            phn = list(map(lambda line: line[2].strip(), lines))

            # if len(phn) < 1:   # if there are no break indices, skip
            #      times = torch.FloatTensor([int(0)])
            if len(phn) < 1:
                return None

        return audio, times.tolist(), phn, spectral_len,  wav_file


    def _filter_dataset(self):
        files = []
        wavs = list(iter_find_files(self.wav_path, "*.wav"))
        for wav in tqdm(wavs, desc="Loading data"):
            res = self.process_file(wav)
            if res is not None:
                files.append(wav)
        return files

    def __getitem__(self, index):
        audio, seg, phn, spectral_len, wav_file = self.process_file(self.data[index])
        return audio, seg, phn, spectral_len, wav_file

    def __len__(self):
        return len(self.data)


    @staticmethod
    def get_dataset(config):
        dataset = phoneme_seg_dataset(config)

        train, test = train_test_split(dataset, test_size=0.2, random_state=42)
        val, test = train_test_split(test, test_size=0.5)
        print("Train: {}, Val: {}, Test: {}".format(len(train), len(val), len(test)))

        return train, val, test


if __name__ ==  "__main__":
    import yaml
    from collections import namedtuple

    with open("../conf/unsupervised_phoneme_seg.yaml", 'r') as f:
        config = yaml.safe_load(f)
    config = namedtuple("config", config.keys())(*config.values())

    train, val, test = phoneme_seg_dataset.get_dataset(config)
    print(train[1])

    train_loader = DataLoader(train, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn_padd)
    print(len(train_loader))
    print(next(iter(train_loader)))
