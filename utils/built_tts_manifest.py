import os
import json
import sox
from tqdm import tqdm
from collections import defaultdict


def build_manifest(TEXT_PATH, WAV_PATH, LAB_PATH):
    files = defaultdict(dict)

    for file in sorted(os.listdir(TEXT_PATH)):
        filename, file_extension = os.path.splitext(os.path.basename(file))
        if file_extension == '.txt':
            with open(TEXT_PATH + file, 'r') as f:
                files[filename]['text'] = f.read().split('\n')[0].split('﻿')[1]

    for file in sorted(os.listdir(LAB_PATH)):
        filename, file_extension = os.path.splitext(os.path.basename(file))
        if file_extension == '.lbl':
            lbl = open(LAB_PATH + file).readlines()
            label_only = [x.split()[1][0] for x in lbl]
            files[filename]['phoneme'] = ' '.join(label_only)
            # with open(LAB_PATH + file, 'r') as f:
                # label_only = [x for x in f.read().split() if x.replace('.', '').isdigit() == False]
                # files[filename]['phoneme'] = ' '.join(label_only)

    data = []

    for filename, content in files.items():
        wav_path = WAV_PATH + filename + '.wav'
        text = content['text']
        phoneme = content['phoneme']
        duration = sox.file_info.duration(wav_path)
        manifest = {
            'audio_filepath': wav_path,
            'duration': duration,
            'text': phoneme,
            # 'normalized_text': phoneme,
            # 'speaker': filename,
        }
        data.append(manifest)

    return data

def save_to_json(dest_path, data, filename):
    if filename[-4:] != 'json':
        filename += '.json'

    with open(os.path.join(dest_path, filename), 'w', encoding='utf-8') as f:
        for line in data:
            json.dump(line, f, ensure_ascii=False)
            f.write('\n')

def main():
    TEXT_PATH = "../script/"
    WAV_PATH = "../wav/"
    LAB_PATH = "../lab/"

    data = build_manifest(TEXT_PATH, WAV_PATH, LAB_PATH)
    save_to_json('../processed', data, 'manifest_tts.json')

if __name__ == '__main__':
    main()
