import sys
import time

sys.path.append('TTS/vits')

import soundfile
import os
import onnxruntime as ort
import json
import numpy as np



import logging
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)


def gen_sym():
    _pad = '_'
    _punctuation = '，。！？—…「」'
    _letters = 'ㄅㄆㄇㄈㄉㄊㄋㄌㄍㄎㄏㄐㄑㄒㄓㄔㄕㄖㄗㄘㄙㄚㄛㄜㄝㄞㄟㄠㄡㄢㄣㄤㄥㄦㄧㄨㄩˉˊˇˋ˙ '
    return [_pad] + list(_punctuation) + list(_letters)
_symbols=gen_sym()
_symbol_to_id = {s: i for i, s in enumerate(_symbols)}
_id_to_symbol = {i: s for i, s in enumerate(_symbols)}
import cleaners
def get_symbols_from_json(path):
    import os
    assert os.path.isfile(path)
    with open(path, 'r') as f:
        data = json.load(f)
    return data['symbols']
def _clean_text(text, cleaner_names):
  for name in cleaner_names:
    cleaner = getattr(cleaners, name)
    if not cleaner:
      raise Exception('Unknown cleaner: %s' % name)
    text = cleaner(text)
  return text

def intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def text_to_sequence(text, cleaner_names):
  sequence = []
  clean_text = _clean_text(text, cleaner_names)
  for symbol in clean_text:
    if symbol not in _symbol_to_id.keys():
      continue
    symbol_id = _symbol_to_id[symbol]
    sequence += [symbol_id]
  return sequence
def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = intersperse(text_norm, 0)
    text_norm = x = np.array([text_norm], dtype=np.int64)
    return text_norm

class HParams():
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()

def get_hparams_from_file(config_path):
    with open(config_path, "r") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    return hparams
class TTService():
    def __init__(self, cfg, model, char, speed):
        logging.info('Initializing TTS Service for %s...' % char)
        self.hps = get_hparams_from_file(cfg)
        self.speed = speed
        self.net_g = ort.InferenceSession(model)

    def read(self, text):
        text = text.replace('~', '！')
        x = get_text(text, self.hps)
        x_len = np.array([x.shape[1]], dtype=np.int64)
        noise = 0.667
        length = 1.0
        noisew = 1.0
        scales = np.array([noise, length, noisew], dtype=np.float32)
        # scales = scales[np.newaxis, :]
        # scales.reshape(1, -1)
        scales.resize(1, 3)

        ort_inputs = {
            'input': x,
            'input_lengths': x_len,
            'scales': scales,
        }
        audio = np.squeeze(self.net_g.run(None, ort_inputs))
        audio *= 32767.0 / max(0.01, np.max(np.abs(audio))) * 0.6
        audio = np.clip(audio, -32767.0, 32767.0)
        return audio

    def read_save(self, text, filename, sr):
        stime = time.time()
        au = self.read(text)
        soundfile.write(filename, au, sr)
        logging.info('VITS Synth Done, time used %.2f' % (time.time() - stime))