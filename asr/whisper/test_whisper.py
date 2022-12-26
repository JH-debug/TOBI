import whisper
import os
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import jiwer


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default='cuda:1', type=str)
    args = parser.parse_args()

    return args


def read_manifest(path):
    manifest = []
    with open(path, 'r') as f:
        for line in tqdm(f, desc='Reading manifest'):
            line = line.replace('\n', '')
            manifest.append(json.loads(line))

    return manifest


def inference(manifest_path, model, transcribe_options):
    grapheme_manifest_path = os.path.join(manifest_path, 'manifest_test_grapheme.json')
    grapheme_manifest_data = read_manifest(grapheme_manifest_path)
    phoneme_manifest_path = grapheme_manifest_path.replace('grapheme', 'phoneme')
    phoneme_manifest_data = read_manifest(phoneme_manifest_path)

    audio_filepaths = []
    grapheme = []
    phoneme = []
    transcriptions = []

    for x, y in tqdm(zip(grapheme_manifest_data, phoneme_manifest_data), desc='Inference'):
        audio_filepath = x['audio_filepath']
        # audio = torch.from_numpy(wavfile.read(audio_filepath)[1].copy())

        audio = whisper.load_audio(audio_filepath)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)

        hypothesis = model.decode(mel, transcribe_options)

        audio_filepaths.append(audio_filepath)
        transcriptions.append(hypothesis.text)
        grapheme.append(x['text'])
        phoneme.append(y['text'])

    data = pd.DataFrame(dict(audio_filepath=audio_filepaths, grapheme=grapheme, phoneme=phoneme, hypothesis=transcriptions))
    data.to_csv(f'result/whisper_test_result.csv', index=False, encoding="utf-8-sig")

    wer = jiwer.wer(list(data["grapheme"]), list(data["hypothesis"]))
    print(f"grapheme WER: {wer * 100:.2f} %")
    wer = jiwer.wer(list(data["phoneme"]), list(data["hypothesis"]))
    print(f"phoneme WER: {wer * 100:.2f} %")


def main():
    args = parse_args()

    print("Loading model...")
    model = whisper.load_model("base", device=args.cuda)
    print(
        f"Model is {'multilingual' if model.is_multilingual else 'English-only'} "
        f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
    )

    options = dict(language="Korean", beam_size=5)
    transcribe_options = whisper.DecodingOptions(fp16=False, task='transcribe', without_timestamps=True, **options)

    if not os.path.isdir('result'):
        os.mkdir('result')

    manifest_path = '../../processed/'
    inference(manifest_path, model, transcribe_options)


if __name__ == '__main__':
    main()