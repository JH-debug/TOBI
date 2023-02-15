# https://github.com/cageyoko/CTC-Attention-Mispronunciation/blob/master/egs/attention_aug/local/make_spectrum.py

import torchaudio


path = 'wav/lmy00001.wav'

sound, _ = torchaudio.load(path)
sound = sound.numpy()

print(sound)
print(sound.mean(axis=1))

windows = {'hamming': torchaudio.functional.create_hamming_window,
           'hann': torchaudio.functional.create_hann_window,
           'blackman': torchaudio.functional.create_blackman_window,
           'bartlett': torchaudio.functional.create_bartlett_window}

audio_conf = {'sample_rate': 16000, 'window_size': 0.025, 'window_stride': 0.01, 'window': 'hamming'}


