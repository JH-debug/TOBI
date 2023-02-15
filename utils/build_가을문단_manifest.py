import os
import json
from tqdm import tqdm
from collections import defaultdict

from utils.conv_code import conv_hangul_to_code


path = "../004_가을문단/"

dataset = []

for level in (0, 1, 2):
    file_path = os.path.join(path, str(level))
    json_path = os.path.join(path, str(level), 'json/')

    for file in os.listdir(json_path):
        filename, file_extension = os.path.splitext(os.path.basename(file))
        if file_extension == '.json':
            with open(json_path + filename + '.json', "r") as f:
                data = json.load(f)

            for i, annotation in enumerate(data['annotations']):
                text_raw = annotation['pronunciationForm']
                start = annotation['start']
                end = annotation['end']

                text_raw = text_raw.replace(' ', '')
                text_raw = text_raw.replace('.', '')
                text_raw = text_raw.replace('깎아', '까까')
                text_raw = text_raw.replace('꺾어', '꺼꺼')
                text_raw = text_raw.replace('놓은', '노은')
                text_raw = text_raw.replace('없', '업')
                text_raw = text_raw.replace('붉은', '불근')
                text_raw = text_raw.replace('숲속', '숩쏙')
                text_raw = text_raw.replace('높은', '노픈')
                text_raw = text_raw.replace('그렇다', '그러타')
                text_raw = text_raw.replace('놓을', '노을')
                text_raw = text_raw.replace('놓고', '노코')
                text_raw = text_raw.replace('좋은', '조은')
                phoneme_text = text_raw.replace('높고', '놉고')

                phoneme_code = conv_hangul_to_code(phoneme_text)

                import wave
                # split the audio file from start to end
                with wave.open(file_path + '/' + filename + '.wav', "rb") as infile:
                    # get file data
                    nchannels = infile.getnchannels()
                    sampwidth = infile.getsampwidth()
                    framerate = infile.getframerate()
                    # set position in wave to start of segment
                    infile.setpos(int(start * framerate))
                    # extract data
                    data = infile.readframes(int((end - start) * framerate))

                wav_path = file_path + '/' + 'processed_wav/' + filename + '_' + str(i + 1) + '.wav'

                # write the extracted data to a new file
                with wave.open(wav_path, 'w') as outfile:
                    outfile.setnchannels(nchannels)
                    outfile.setsampwidth(sampwidth)
                    outfile.setframerate(framerate)
                    outfile.setnframes(int(len(data) / sampwidth))
                    outfile.writeframes(data)

                manifest = {
                    "severity": level,
                    'audio_filepath': wav_path,
                    'phoneme_text': phoneme_text,
                    'phoneme_code': phoneme_code,
                    'index': i+1,
                     # 'start': start,
                     # 'end': end,
                }

                dataset.append(manifest)

with open(f'../processed/가을문단.json', 'w', encoding='UTF-8') as f:
    for line in dataset:
        json.dump(line, f, ensure_ascii=False)
        f.write('\n')
