"""Реплики Смауга из «Пустоши Смауга» и «Битвы пяти воинств» (обе Extended).

WavLM-sv почти не отличает Смауга от Торина и других низких мужских голосов, а фразы whisper часто
склеивают реплики двух персонажей. Поэтому границы и текст берутся из субтитров (одна строка = одна реплика
одного персонажа), а реплики Смауга размечены вручную по индексам строк SRT.

    .venv/bin/python scripts/smaug_from_srt.py path/to/eng.srt

1. подряд идущие строки Смауга (пауза < MAX_GAP) делятся на фразы <= MAX_DUR динамическим программированием:
   резать лучше на конце предложения, фразы короче 2 с нежелательны
   В «Битве» Смауг говорит только в первые 10 минут, субтитров к ней нет: таймкоды взяты из whisper (BOFA ниже)
2. отрезок берётся из вокала demucs; края: начало строки - 0.2 с, конец + 0.3 с, но не заходя на чужие строки;
   тишина по краям обрезается
3. проверка: повторная расшифровка whisper и WER относительно текста субтитров; WavLM-близость к эталонам
Результат: data/segments/smaug/wavs/*.wav (22050 Гц), segments.csv в формате prep_data.py
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRT = sys.argv[1]
RAW = os.path.join(ROOT, 'data', 'raw', 'smaug')
SEP = os.path.join(ROOT, 'data', 'separated', 'smaug', 'htdemucs')
OUT = os.path.join(ROOT, 'data', 'segments', 'smaug')
SEED = os.path.join(ROOT, 'data', 'seed', 'smaug')
SR = 22050
MAX_GAP, MAX_DUR = 2.5, 11.0  # Смауг говорит с длинными паузами
DEVICE = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')

# номера строк SRT, которые произносит Смауг (сцена в Эреборе 2:17-2:56)
SMAUG = [*range(1868, 1886), *range(1887, 1890), 1897, 1902, *range(1905, 1910), 1911, 1916, 1917, 1921, 1922,
         *range(1924, 1928), *range(1932, 1936), *range(1939, 1954), *range(1973, 1982), *range(2019, 2023),
         *range(2025, 2031), *range(2035, 2042), *range(2045, 2057), 2059, *range(2064, 2077), 2112,
         *range(2170, 2175), *range(2229, 2240), *range(2244, 2249), *range(2253, 2264), 2271, 2273, 2274,
         *range(2278, 2281)]
# «Битва»: (начало, конец) в секундах от начала фильма, текст; по таймкодам слов whisper
BOFA = [(546.2, 551.3, 'Who are you that would stand against me?'),
        (555.0, 557.9, 'Now, that is a pity.'),
        (559.3, 564.4, 'What will you do now, bowman?'),
        (565.3, 571.9, 'You are forsaken. No help will come.'),
        (580.9, 589.85, 'Is that your child? You cannot save him from the fire.'),
        (589.85, 592.7, 'He will burn.'),
        (616.2, 624.6, 'Tell me, wretch. How now shall you challenge me?'),
        (628.6, 635.6, 'You have nothing left but your death.')]
TEXT_FIX = {2245: 'do you?', 2019: 'So, Oakenshield...', 2229: 'You think you could...', 2230: '...deceive me, Barrel-rider?'}  # сверено со звуком


def read_srt(path):
    def t(s):
        h, m, r = s.strip().split(':')
        s, ms = r.split(',')
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000
    items = []
    for b in re.split(r'\n\s*\n', open(path, encoding='utf-8-sig').read().strip()):
        L = b.strip().split('\n')
        if len(L) < 3 or '-->' not in L[1]:
            continue
        a, c = L[1].split('-->')
        text = re.sub(r'<[^>]+>', '', ' '.join(L[2:])).strip()
        text = re.sub(r'^[A-Z ]+:\s*', '', text)  # метка говорящего «SMAUG: »
        i = int(L[0])
        items.append(dict(i=i, a=t(a), c=t(c), text=TEXT_FIX.get(i, text)))
    return items


def clean_join(texts):
    s = ' '.join(texts)
    s = re.sub(r'\.\.\.\s*\.\.\.', ' ', s)  # «Something you... ...carry» -> «Something you carry»
    s = re.sub(r'(^|\s)\.\.\.', r'\1', s)
    s = re.sub(r'\.\.\.(?=\s+[a-z])', '', s)  # «You care about them... do you?»
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def norm_words(text):
    return re.sub(r"[^a-z' ]", ' ', text.lower().replace('-', ' ')).split()


def wer(ref, hyp):
    r, h = norm_words(ref), norm_words(hyp)
    d = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        prev, d[0] = d.copy(), i
        for j in range(1, len(h) + 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(h)] / max(1, len(r))


def split_run(r):
    """DP: фразы <= MAX_DUR; штраф за разрез не на конце предложения и за фразы короче 2 с"""
    n = len(r)
    end_sent = [bool(re.search(r'[.!?]$', x['text'])) and not x['text'].endswith('...') for x in r]
    best, back = [0.0] + [np.inf] * n, [0] * (n + 1)
    for j in range(1, n + 1):
        for i in range(j):
            dur = r[j - 1]['c'] - r[i]['a']
            if dur > MAX_DUR and j - i > 1:
                continue
            cost = best[i] + 1 + 3 * (j < n and not end_sent[j - 1]) + 3 * (dur < 2.0)
            if cost < best[j]:
                best[j], back[j] = cost, i
    out, j = [], n
    while j:
        out.append(r[back[j]:j])
        j = back[j]
    return out[::-1]


def main():
    from faster_whisper import WhisperModel
    from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

    items = read_srt(SRT)
    smaug = set(SMAUG)
    other = [x for x in items if x['i'] not in smaug and not x['text'].startswith('(')]  # чужие реплики

    runs = []
    for x in [x for x in items if x['i'] in smaug]:
        if runs and x['i'] - runs[-1][-1]['i'] == 1 and x['a'] - runs[-1][-1]['c'] < MAX_GAP:
            runs[-1].append(x)
        else:
            runs.append([x])
    groups = [g for r in runs for g in split_run(r)]

    def vocals(prefix, lo, hi):
        """вокал demucs кусков фильма, покрывающих [lo, hi] (секунды от начала фильма), одной дорожкой"""
        files = sorted(f for f in os.listdir(RAW) if f.startswith(prefix) and f.endswith('.wav'))
        offs, t0 = [], 0.0
        for f in files:
            d = sf.info(os.path.join(RAW, f)).duration
            offs.append((f, t0, t0 + d))
            t0 += d
        need = [(f, s0) for f, s0, s1 in offs if s1 > lo and s0 < hi]
        y = np.concatenate([librosa.load(os.path.join(SEP, f[:-4], 'vocals.wav'), sr=SR, mono=True)[0]
                            for f, _ in need])
        return y, need[0][1]

    # кандидаты: (фильм, начало, конец, текст, номера строк SRT)
    cands = []
    for g in groups:
        a, b = g[0]['a'] - 0.2, g[-1]['c'] + 0.3
        prev_end = max([o['c'] for o in other if o['c'] <= g[0]['a'] + 0.1], default=-1)
        next_start = min([o['a'] for o in other if o['a'] >= g[-1]['c'] - 0.1], default=1e9)
        cands.append(('dos', max(a, prev_end + 0.05), min(b, next_start - 0.05), clean_join([x['text'] for x in g]),
                      ','.join(str(x['i']) for x in g)))
    cands += [('bofa', a, b, t, '') for a, b, t in BOFA]
    audio = {}
    for film, prefix in (('dos', 'film_dos_ext'), ('bofa', 'film_bofa_ext')):
        ts = [c[1] for c in cands if c[0] == film] + [c[2] for c in cands if c[0] == film]
        audio[film] = vocals(prefix, min(ts) - 5, max(ts) + 5)
        print(film, 'вокал с', round(audio[film][1], 1), 'с, длина', round(len(audio[film][0]) / SR, 1), 'с',
              flush=True)

    asr = WhisperModel('large-v3', device=DEVICE, compute_type='float16')
    fe = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base-sv')
    sv = WavLMForXVector.from_pretrained('microsoft/wavlm-base-sv').eval().to(DEVICE)

    def embed(x):
        with torch.no_grad():
            e = sv(**fe(librosa.resample(x, orig_sr=SR, target_sr=16000), sampling_rate=16000,
                         return_tensors='pt').to(DEVICE)).embeddings
        return torch.nn.functional.normalize(e, dim=-1)[0].cpu().numpy()

    seeds = [os.path.join(SEED, s) for s in sorted(os.listdir(SEED)) if s.endswith('.wav')]
    c = np.mean([embed(librosa.load(s, sr=SR, mono=True)[0]) for s in seeds], 0)
    c /= np.linalg.norm(c)

    os.makedirs(os.path.join(OUT, 'wavs'), exist_ok=True)
    for f in os.listdir(os.path.join(OUT, 'wavs')):
        os.remove(os.path.join(OUT, 'wavs', f))
    rows = []
    for film, a, b, text, srt in cands:
        y, base = audio[film]
        seg = y[int((a - base) * SR):int((b - base) * SR)]
        if len(seg) < SR * 0.5 or np.abs(seg).max() < 1e-3:
            continue
        _, (i0, i1) = librosa.effects.trim(seg, top_db=35, frame_length=1024, hop_length=256)
        i0, i1 = max(0, i0 - int(0.1 * SR)), min(len(seg), i1 + int(0.15 * SR))
        a, b = a + i0 / SR, a + i1 / SR
        seg = seg[i0:i1]
        hyp = ' '.join(s.text for s in asr.transcribe(librosa.resample(seg, orig_sr=SR, target_sr=16000),
                                                      language='en', beam_size=5, vad_filter=False,
                                                      condition_on_previous_text=False)[0]).strip()
        seg = seg / np.abs(seg).max() * 0.9
        sid = f'{film}_{int(a * 100):07d}'
        sf.write(os.path.join(OUT, 'wavs', sid + '.wav'), seg, SR)
        rows.append(dict(id=sid, source=film, start=round(a, 2), end=round(b, 2), dur=round(b - a, 2),
                         text=text, dup=False, srt=srt, asr=hyp,
                         wer=round(wer(text, hyp), 3), spk_sim=float(embed(seg) @ c)))
        print(f"{sid} {b - a:5.2f}s wer={rows[-1]['wer']:.2f} sim={rows[-1]['spk_sim']:.3f} | {text} | {hyp}",
              flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'segments.csv'), index=False)
    print(json.dumps(dict(n=len(df), minutes=round(df.dur.sum() / 60, 2),
                          minutes_wer_lt_0_5=round(df[df.wer < 0.5].dur.sum() / 60, 2))))


if __name__ == '__main__':
    main()
