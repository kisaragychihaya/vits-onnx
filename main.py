import json
import onnxruntime as ort
import cleaners
import numpy as np
from scipy.io import wavfile
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

def gen_sym():
    _pad = '_'
    _punctuation = '，。！？—…「」'
    _letters = 'ㄅㄆㄇㄈㄉㄊㄋㄌㄍㄎㄏㄐㄑㄒㄓㄔㄕㄖㄗㄘㄙㄚㄛㄜㄝㄞㄟㄠㄡㄢㄣㄤㄥㄦㄧㄨㄩˉˊˇˋ˙ '
    return [_pad] + list(_punctuation) + list(_letters)
_symbols=gen_sym()
_symbol_to_id = {s: i for i, s in enumerate(_symbols)}
_id_to_symbol = {i: s for i, s in enumerate(_symbols)}
def get_hparams_from_file(config_path):
    with open(config_path, "r") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    return hparams

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
def text_to_sequence(text, cleaner_names):
  sequence = []
  clean_text = _clean_text(text, cleaner_names)
  for symbol in clean_text:
    if symbol not in _symbol_to_id.keys():
      continue
    symbol_id = _symbol_to_id[symbol]
    sequence += [symbol_id]
  return sequence

def intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result

cfg="paimon6k.json"
symbols = get_symbols_from_json(cfg)
hps=get_hparams_from_file(cfg)
text="  旅行者，我们是变成了数字生命了吗？"
ort_sess = ort.InferenceSession("paimon-tts.onnx")
seq = text_to_sequence(text, cleaner_names=hps.data.text_cleaners)
if hps.data.add_blank:
    seq = intersperse(seq, 0)

x = np.array([seq], dtype=np.int64)
x_len = np.array([x.shape[1]], dtype=np.int64)
# sid = np.array([sid], dtype=np.int64)
# noise(可用于控制感情等变化程度) lenth(可用于控制整体语速) noisew(控制音素发音长度变化程度)
                # 参考 https://github.com/gbxh/genshinTTS

noise=0.667
length=1.0
noisew=1.0
scales = np.array([noise, length, noisew], dtype=np.float32)
                # scales = scales[np.newaxis, :]
                # scales.reshape(1, -1)
scales.resize(1, 3)

ort_inputs = {
                'input': x,
                'input_lengths': x_len,
                'scales': scales,
                }


import time
                # start_time = time.time()
start_time = time.perf_counter()
audio = np.squeeze(ort_sess.run(None, ort_inputs))
audio *= 32767.0 / max(0.01, np.max(np.abs(audio))) * 0.6
audio = np.clip(audio, -32767.0, 32767.0)
end_time = time.perf_counter()
# end_time = time.time()
print("infer time cost: ", end_time - start_time, "s")

wavfile.write("out.wav",
                hps.data.sampling_rate, audio.astype(np.int16))