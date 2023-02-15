import os
import json
import sox
from tqdm import tqdm


WAV_PATH = "/home/jhlee/TOBI/wav/"

def build_manifest(split='grapheme'):
    if split == 'grapheme':
        split = 'text'
    else:
        split = 'phoneme'

    data = []
    with open("../processed/seq2seq_test.json", "r") as f:
        test = json.load(f)
        for content in tqdm(test):
            if content:
                wav_path = WAV_PATH + content['id'] + '.wav'
                text = content[split]
                duration = sox.file_info.duration(wav_path)

                manifest = {
                    'audio_filepath': wav_path,
                    'duration': duration,
                    'text': text
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
    data = build_manifest(split='grapheme')
    save_to_json('../processed', data, 'manifest_test_grapheme.json')
    data = build_manifest(split='phoneme')
    save_to_json('../processed', data, 'manifest_test_phoneme.json')

if __name__ == '__main__':
    main()
