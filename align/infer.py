import argparse
import json
import os
from tqdm import tqdm

import librosa
import soundfile as sf
import torch

from nemo.collections.tts.models import AlignerModel
from nemo.collections.tts.torch.helpers import general_padding

import sys
sys.path.append('..')
from utils.conv_code import conv_hangul_to_code, conv_code_to_phoneme, conv_code_to_phoneme_with_duration


def get_args():
    """Retrieve arguments for disambiguation.
    """
    parser = argparse.ArgumentParser("G2P disambiguation using Aligner input embedding distances.")
    parser.add_argument('--model',
                        required=True,
                        type=str,
                        help="Path to Aligner model checkpoint (.nemo file)."
    )
    parser.add_argument('--manifest',
                        required=True,
                        type=str,
                        help="Path to data manifest. Each entry should contain the path to the audio file as well as the text in graphemes.",
    )
    parser.add_argument('--out',
                        required=True,
                        type=str,
                        help="Path to output file where file will be written."
    )
    parser.add_argument('--sr',
                        required=False,
                        default=44100,
                        type=int,
                        help="Target sample rate to load the dataset. Should match what the model was trained on.",
    )
    args = parser.parse_args()
    return args


def load_and_prepare_audio(aligner, audio_path, target_sr, device):
    """Loads and resamples audio to target sample rate (if necessary), and preprocesses for Aligner input.
    """
    # Load audio and get length for preprocessing
    audio_data, orig_sr = sf.read(audio_path)
    if orig_sr != target_sr:
        audio_data = librosa.core.resample(audio_data, orig_sr=orig_sr, target_sr=target_sr)

    audio = torch.tensor(audio_data, dtype=torch.float, device=device).unsqueeze(0)
    audio_len = torch.tensor(audio_data.shape[0], device=device).long().unsqueeze(0)

    # Generate spectrogram
    spec, spec_len = aligner.preprocessor(input_signal=audio, length=audio_len)

    return audio, audio_len, spec, spec_len


def aligner_predict(aligner, text, spec, spec_len, device):

    aligner_tokenizer = aligner.tokenizer

    # convert text to phoneme code
    text_raw = conv_hangul_to_code(text)

    # tokenize text
    text_encode = aligner_tokenizer.encode(text_raw)
    text_len = len(text_encode)

    text = torch.tensor(text_encode, device=device).unsqueeze(0).long()
    text_len = torch.tensor(text_len, device=device).unsqueeze(0).long()

    # prediction
    with torch.no_grad():
        attn_soft_tensor, attn_logprob_tensor = aligner(spec=spec, spec_len=spec_len, text=text, text_len=text_len)

    attn_soft = attn_soft_tensor[0, 0, :, :].data.cpu().numpy()
    attn_logprob = attn_logprob_tensor[0, 0, :, :].data.cpu().numpy()

    durations = aligner.alignment_encoder.get_durations(attn_soft_tensor, text_len, spec_len).int()
    # text_g2p = [x for x in text_raw if x != ' ']

    # dictionary = {t: d for t, d in zip(text_g2p, durations.cpu().numpy()[0])}

    # find the most probable phoneme sequence with break index
    processed_text = text_raw.replace(' ', '')
    text_list, duration_list = conv_code_to_phoneme_with_duration(processed_text, durations.cpu().numpy()[0].tolist())

    split_duration = duration_list[round(len(text_list) * 0.1):round(len(text_list) * 0.9)]
    split_duration = torch.tensor(split_duration, dtype=torch.float64)
    std, mean = torch.std_mean(split_duration, unbiased=False)
    # median = torch.median(split_duration)

    k = round(len(duration_list) * 0.1)
    top_k_values, top_k_indicies = torch.topk(split_duration, k)

    indices = []
    for i, (index, value) in enumerate(zip(top_k_indicies, top_k_values)):
        if i == 0 and k == 1:
            if len(str(value)) > 2:
                indices.append(index)
        elif i == 0 and k > 1:
            if len(str(value)) > len(str(top_k_values[i + 1])):
                indices.append(index)
        else:
            if abs(index - top_k_indicies[i - 1]) == 1:
                if value and top_k_values[i - 1] > mean + std:
                    indices.append(index)

    for ind in indices:
        text_list.insert(round(len(text_list) * 0.1) + ind+1, ';')

    predicted_text = ' '.join(text_list)
    predicted_phoneme = conv_hangul_to_code(''.join(text_list))

    return durations, predicted_text, predicted_phoneme


def Inference_dataset(
    aligner, manifest_path, out_path, sr, device):

    with open(out_path, 'w',  encoding='utf-8') as f_out:
        with open(manifest_path, 'r') as f_in:
            count = 0

            for line in tqdm(f_in):
                # Retrieve entry and base G2P conversion for full text
                entry = json.loads(line)
                # Set punct_post_process=True in order to preserve words with apostrophes
                text = entry['phoneme_text']

                # Load and preprocess audio
                audio_path = entry['audio_filepath']
                audio, audio_len, spec, spec_len = load_and_prepare_audio(aligner, audio_path, sr, device)

                # Get durations
                durations, predicted_text, predicted_phoneme = aligner_predict(
                    aligner, text, spec, spec_len, device
                )

                entry['predicted_duration_per_phoneme'] = durations.cpu().numpy()[0].tolist()
                entry['predicted_text'] = predicted_text
                entry['predicted_phoneme'] = predicted_phoneme

                # print("original text: ", text)
                # print("prediction   : ", predicted_text)

                f_out.write(f"{json.dumps(entry, ensure_ascii=False)}\n")

                count += 1
                if count % 100 == 0:
                    print(f"Finished {count} entries.")

    print(f"Finished all entries, with a total of {count}.")


def main():
    args = get_args()
    # Check file paths from arguments
    if not os.path.exists(args.model):
        print("Could not find model checkpoint file: ", args.model)
    if not os.path.exists(args.manifest):
        print("Could not find data manifest file: ", args.manifest)
    if os.path.exists(args.out):
        print("Output file already exists: ", args.out)
        overwrite = input("Is it okay to overwrite it? (Y/N): ")
        if overwrite.lower() != 'y':
            print("Not overwriting output file, quitting.")
            quit()

    # Load model
    print("Restoring Aligner model from checkpoint...")
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    if args.model.endswith('.nemo'):
        aligner = AlignerModel.restore_from(args.model, map_location=device)
    else:
        aligner = AlignerModel.load_from_checkpoint(args.model).to(device)

    # Inference
    print("Beginning Inference...")
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    Inference_dataset(aligner, args.manifest, args.out, args.sr, device)


if __name__ == '__main__':
    main()