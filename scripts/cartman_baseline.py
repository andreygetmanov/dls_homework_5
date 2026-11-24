"""База для критерия «голос похож» (дополнительный раздел): similarity трёх тестовых референсов Картмана
с другими фразами Картмана, с другими детьми той же озвучки (Стэн, Кайл, Венди), со взрослыми персонажами
и с чужим голосом (тестовые референсы Смауга). Протокол как в основной части: каждый референс с каждой фразой.

    .venv/bin/python scripts/cartman_baseline.py
Результат: outputs/cartman_baseline.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import soundfile as sf
import librosa

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xtts_eval as xe

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LJS = os.path.join(ROOT, 'data', 'ljs', 'cartman')
SEG = os.path.join(ROOT, 'data', 'segments', 'cartman')
SEP = os.path.join(ROOT, 'data', 'separated', 'cartman', 'htdemucs')
TMP = os.path.join(ROOT, 'data', 'baseline', 'cartman')
SR = 22050
N = 30  # фраз на группу
rng = np.random.default_rng(0)


def main():
    refs = [os.path.join(LJS, l.strip()) for l in open(os.path.join(LJS, 'test_refs.txt')) if l.strip()]
    split = pd.read_csv(os.path.join(LJS, 'split.csv'))
    cart = [os.path.join(LJS, 'wavs', i + '.wav') for i in
            rng.choice(split[split.split == 'eval'].id.tolist() + split[split.split == 'full'].id.tolist(), N,
                       replace=False)]
    od = pd.read_csv(os.path.join(SEG, 'others.csv'))
    od = od[(od.end - od.start) >= 2.0]
    groups = {'kids': od[od.who.isin(['stan', 'kyle', 'wendy'])],
              'adults': od[od.who.isin(['mr. garrison', 'chef', 'jimbo', 'mayor', 'jesus', 'officer barbrady'])]}
    os.makedirs(TMP, exist_ok=True)
    paths = {}
    cache = {}
    for g, d in groups.items():
        d = d.iloc[rng.choice(len(d), N, replace=False)]
        paths[g] = []
        for r in d.itertuples():
            if r.source not in cache:
                cache[r.source] = librosa.load(os.path.join(SEP, r.source + '_mtv', 'vocals.wav'), sr=SR)[0]
            y = cache[r.source][int(r.start * SR):int(r.end * SR)]
            p = os.path.join(TMP, f'{g}_{r.source}_{int(r.start * 100):06d}.wav')
            sf.write(p, y / np.abs(y).max() * 0.9, SR)
            paths[g].append(p)
    smaug = os.path.join(ROOT, 'data', 'ljs', 'smaug')
    paths['smaug'] = [os.path.join(smaug, l.strip()) for l in open(os.path.join(smaug, 'test_refs.txt')) if l.strip()]
    paths['cartman'] = cart
    sv = xe.SpeakerSim()
    gt = sv.refs_pairwise(refs)
    res = {'gt_pairwise': dict(mean=float(gt.mean()), std=float(gt.std()))}
    for g in ('cartman', 'kids', 'adults', 'smaug'):
        M = sv.matrix(refs, paths[g])
        res[g] = dict(n=len(paths[g]), mean=float(M.mean()), std=float(M.std()),
                      share_gt_086=float((M > 0.86).mean()))
        print(g, f'{M.mean():.4f} ± {M.std():.4f}  >0.86: {(M > 0.86).mean():.2f}', flush=True)
    json.dump(res, open(os.path.join(ROOT, 'outputs', 'cartman_baseline.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
