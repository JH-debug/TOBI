import numpy as np
import soundfile as sf
from boltons.fileutils import iter_find_files
from tqdm import tqdm

wav_path = '/home/jhlee/TOBI/wav/'
phn_path = '/home/jhlee/TOBI/lab/'

wavs = list(iter_find_files(wav_path, "*.wav"))

for wav_file in tqdm(wavs):
    wav, sr = sf.read(wav_file)
    phn_file = wav_file.replace(wav_path, phn_path).replace(".wav", ".lbl")
    with open(phn_file, 'r') as f:
        phn = f.readlines()
        phn = list(map(lambda x: x.replace('\n', '').split('\t'), phn))
        time = list(map(lambda phn: phn[0].strip(), phn))
        phn = list(map(lambda phn: phn[1].strip(), phn))

    delim_locations = np.array([i for i, phone in enumerate(phn)])
    segment = []

    for i, (t,p) in enumerate(zip(time, phn)):
        if i == 0:
            segment_start_time = 0
            segment_end_time = float(t)
            segment.append([segment_start_time, segment_end_time, p])
        elif 1 <= i <= len(delim_locations) - 1:
            segment_start_time = float(time[i-1])
            segment_end_time = float(time[i])
            segment.append([segment_start_time, segment_end_time, p])

    segment_start_time = 0
    segment_end_time = float(time[-1])
    phn_data = "\n".join(
        [f"{int((p[0] - segment_start_time) * sr)} {int((p[1] - segment_start_time) * sr)} {p[2]}" for p in segment])
    with open(phn_file.replace('.lbl', '.phone'), 'w') as f:
        f.write(phn_data)


# import buckeye
#
# wav = 's3403b.wav'
# name = wav.replace('.wav', '')
# words = wav.replace('.wav', '.words')
# phones = wav.replace('.wav', '.phones')
# log = wav.replace('.wav', '.log')
# txt = wav.replace('.wav', '.txt')
#
# track = buckeye.Track(name=name,
#                       words=words,
#                       phones=phones,
#                       log=log,
#                       txt=txt,
#                       wav=wav)
#
# DELIMITER = ['VOCNOISE', 'NOISE', 'SIL']
# FORBIDDEN = ['{B_TRANS}', '{E_TRANS}', '<EXCLUDE-name>', 'LAUGH', 'UNKNOWN', 'IVER-LAUGH', '<exclude-Name>', 'IVER']
# NOISE_EDGES = 0.2
# is_delim = lambda x: x.seg in DELIMITER
#
# phones = track.phones[1:-1]
# delim_locations = np.array([i for i, phone in enumerate(phones) if is_delim(phone)])
# loaded_wav, sr = sf.read(wav)
#
# for start, end in zip(delim_locations[:-1], delim_locations[1:]):
#     segment = phones[start:end+1]
#     segment_start_time = segment[0].beg
#     segment_end_time = segment[-1].end
#     print(segment_start_time, segment_end_time)
#
#     phn_data = "\n".join(
#         [f"{int((p.beg - segment_start_time) * sr)} {int((p.end - segment_start_time) * sr)} {p.seg}" for p in segment])
#     print(phn_data)