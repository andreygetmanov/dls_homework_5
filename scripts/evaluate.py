"""Оценка трёх моделей по сгенерированным wav (outputs/gens/<model>/<model>_XX.wav).

    .venv/bin/python scripts/evaluate.py smaug zs 1min full
    .venv/bin/python scripts/evaluate.py cartman zs 1min full   # русский: SENTENCES_RU, outputs/gens_cartman

- speaker similarity (WavLM-sv, как на семинаре): GT = попарно между 3 тестовыми референсами,
  модель = каждый из 3 референсов против каждой из 10 генераций (30 пар)
- CER: whisper large-v3 (MPS) на генерации против исходного текста, после нормализации
- секунд на символ текста: ловит обрывы и зацикливание
Результат: outputs/metrics.json
"""
import json
import os
import re
import sys

import numpy as np
import librosa

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xtts_eval as xe

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
name, models = sys.argv[1], sys.argv[2:]
RU = name == 'cartman'
LANG, SENT = ('ru', xe.SENTENCES_RU) if RU else ('en', xe.SENTENCES)
LJS = os.path.join(ROOT, 'data', 'ljs', name)
GENS = os.path.join(ROOT, 'outputs', 'gens_cartman' if RU else 'gens')
OUT = os.path.join(ROOT, 'outputs', 'metrics_cartman.json' if RU else 'metrics.json')
refs = [os.path.join(LJS, l.strip()) for l in open(os.path.join(LJS, 'test_refs.txt')) if l.strip()]


def norm(t):
    return re.sub(r'\s+', ' ', re.sub(r"[^a-zа-я' ]", ' ', t.lower().replace('ё', 'е'))).strip()


def cer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    d = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        prev, d[0] = d.copy(), i
        for j in range(1, len(h) + 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(h)] / max(1, len(r))


def main():
    from faster_whisper import WhisperModel
    sv = xe.SpeakerSim()
    asr = WhisperModel('large-v3', device=xe.default_device() if xe.default_device() != 'cpu' else 'cpu',
                       compute_type='float16')
    gt = sv.refs_pairwise(refs)
    res = {'gt': dict(pairs=gt.round(4).tolist(), mean=float(gt.mean()), std=float(gt.std()))}
    for m in models:
        gens = [os.path.join(GENS, m, f'{m}_{i:02d}.wav') for i in range(len(SENT))]
        M = sv.matrix(refs, gens)
        hyps, cers, spc = [], [], []
        for g, t in zip(gens, SENT):
            y, _ = librosa.load(g, sr=16000)
            hyp = ' '.join(s.text for s in asr.transcribe(y, language=LANG, beam_size=5)[0]).strip()
            hyps.append(hyp)
            cers.append(cer(t, hyp))
            spc.append(len(y) / 16000 / len(t))
        res[m] = dict(sim_mean=float(M.mean()), sim_std=float(M.std()), sim_per_ref=M.mean(1).round(4).tolist(),
                      sim_per_sentence=M.mean(0).round(4).tolist(), cer_mean=float(np.mean(cers)),
                      cer=np.round(cers, 3).tolist(), sec_per_char=np.round(spc, 3).tolist(), asr=hyps)
        print(m, f"sim {M.mean():.4f}±{M.std():.4f}  CER {np.mean(cers):.3f}", flush=True)
    json.dump(res, open(OUT, 'w'), indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
