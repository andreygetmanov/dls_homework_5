"""Генерация XTTS-v2 (zero-shot или дообученный чекпойнт) и speaker similarity на WavLM-sv.

Все три модели генерируются одной функцией с одинаковыми параметрами и сидом,
чтобы различие было только в весах GPT.
"""
import os

import numpy as np
import librosa
import soundfile as sf
import torch

os.environ.setdefault('COQUI_TOS_AGREED', '1')

SENTENCES = [
    "For a full hour he had paced up and down, waiting: but he could wait no longer.",
    "Our mouths open, our lips move, but not a word can be heard.",
    "For a long time he had wished to explore the beautiful Land of Oz in which they lived.",
    "The birch canoe slid on the smooth planks.",
    "It is easy to tell the depth of a well.",
    "The old house at the end of the street has been empty for many years.",
    "Nobody expected the meeting to last until well after midnight.",
    "She carried a heavy basket of apples up the narrow stairs.",
    "If you listen carefully, you can hear the river behind the trees.",
    "We will need more time, more patience, and a great deal of luck.",
]


def default_device():
    if torch.cuda.is_available():
        return 'cuda'
    return 'mps' if torch.backends.mps.is_available() else 'cpu'


def base_model_dir():
    from TTS.utils.manage import ModelManager
    path, _, _ = ModelManager().download_model('tts_models/multilingual/multi-dataset/xtts_v2')
    path = str(path)
    return path if os.path.isdir(path) else os.path.dirname(path)  # в coqui-tts 0.28 возвращается путь к model.pth


def load_xtts(checkpoint=None, device=None, base=None):
    """checkpoint=None -> исходный XTTS-v2; иначе путь к best_model.pth после дообучения.
    base: папка с config.json, vocab.json, model.pth (по умолчанию скачивается через ModelManager)"""
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    base = base or base_model_dir()
    config = XttsConfig()
    config.load_json(os.path.join(base, 'config.json'))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=base, checkpoint_path=checkpoint or os.path.join(base, 'model.pth'),
                          vocab_path=os.path.join(base, 'vocab.json'), use_deepspeed=False)
    device = device or default_device()
    model.to(device)
    model.eval()  # Xtts.eval() возвращает None, поэтому не цепочкой
    return model


@torch.inference_mode()
def synthesize(model, ref_wav, texts, out_dir, prefix, seed=0, lang='en'):
    os.makedirs(out_dir, exist_ok=True)
    gpt_cond, spk = model.get_conditioning_latents(audio_path=[ref_wav])
    paths = []
    for i, t in enumerate(texts):
        torch.manual_seed(seed + i)
        out = model.inference(t, lang, gpt_cond, spk, temperature=0.75, enable_text_splitting=True)
        p = os.path.join(out_dir, f'{prefix}_{i:02d}.wav')
        sf.write(p, np.asarray(out['wav']), 24000)
        paths.append(p)
    return paths


class SpeakerSim:
    """WavLM-sv как на семинаре: x-vector, L2-нормировка, косинус"""

    def __init__(self, device=None):
        device = device or default_device()
        from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
        self.fe = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base-sv')
        self.model = WavLMForXVector.from_pretrained('microsoft/wavlm-base-sv').eval().to(device)
        self.device = device

    @torch.inference_mode()
    def embed(self, path):
        y, _ = librosa.load(path, sr=16000, mono=True)
        x = self.fe(y, sampling_rate=16000, return_tensors='pt').to(self.device)
        return torch.nn.functional.normalize(self.model(**x).embeddings, dim=-1)[0].cpu().numpy()

    def matrix(self, refs, gens):
        """косинусы: строки refs, столбцы gens"""
        R = np.stack([self.embed(p) for p in refs])
        G = np.stack([self.embed(p) for p in gens])
        return R @ G.T

    def refs_pairwise(self, refs):
        R = np.stack([self.embed(p) for p in refs])
        S = R @ R.T
        return S[np.triu_indices(len(refs), 1)]
