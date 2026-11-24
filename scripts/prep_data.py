"""Сырые дорожки -> фразы 1.5-11 с с расшифровкой, фильтром по спикеру и без дублей.

    .venv/bin/python scripts/prep_data.py smaug [--seed data/seed/smaug] [--order film1.wav,film2.wav,...]

1. demucs (htdemucs) отделяет голос от музыки и эффектов
2. faster-whisper (MPS) расшифровывает с таймкодами слов; результат кэшируется в data/asr/<name>/<файл>.json
3. фразы: границы предложений whisper, склейка соседних при паузе < MAX_GAP, длинные режутся на паузах между словами;
   выкидываются куски, где слов мало для их длительности (whisper потерял слова)
4. WavLM-sv эмбеддинг каждой фразы; близость к центроиду спикера. Центроид строится по эталонным записям (--seed),
   если они есть: в фильме много других голосов, медиана по всем фразам там не годится
5. дубли: одна и та же реплика в нескольких исходниках (нарезки из тех же сцен). Исходники обрабатываются по порядку,
   фраза считается дублем, если >= 50% её словесных триграмм уже встречались в расшифровках предыдущих исходников
Результат: data/segments/<name>/wavs/*.wav (22050 Гц, моно), segments.csv, emb.npy
"""
import argparse
import json
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = argparse.ArgumentParser()
p.add_argument('name')
p.add_argument('--seed', default=None, help='папка с эталонными wav этого спикера для центроида')
p.add_argument('--order', default=None, help='префиксы имён исходников через запятую (первые приоритетнее при дублях)')
a = p.parse_args()

NAME = a.name
RAW = os.path.join(ROOT, 'data', 'raw', NAME)
SEP = os.path.join(ROOT, 'data', 'separated', NAME)
ASR = os.path.join(ROOT, 'data', 'asr', NAME)
OUT = os.path.join(ROOT, 'data', 'segments', NAME)
SR = 22050
MIN_DUR, MAX_DUR, MAX_GAP = 1.5, 11.0, 0.5
MIN_CPS = 6.0  # символов в секунду: обычная речь 12-15, меньше 6 значит, что whisper потерял слова
DEVICE = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
for d in (SEP, ASR, os.path.join(OUT, 'wavs')):
    os.makedirs(d, exist_ok=True)


def separate(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    vocals = os.path.join(SEP, 'htdemucs', stem, 'vocals.wav')
    if not os.path.exists(vocals):
        subprocess.run([sys.executable, '-m', 'demucs', '--two-stems', 'vocals', '-n', 'htdemucs',
                        '-d', DEVICE, '-o', SEP, path], check=True)
        no_vocals = os.path.join(SEP, 'htdemucs', stem, 'no_vocals.wav')
        if os.path.exists(no_vocals):
            os.remove(no_vocals)  # аккомпанемент не нужен, а на фильмах это гигабайты
    return vocals


def transcribe(asr, vocals, src):
    cache = os.path.join(ASR, os.path.splitext(src)[0] + '.json')
    if os.path.exists(cache):
        return json.load(open(cache))
    y16, _ = librosa.load(vocals, sr=16000, mono=True)  # массивом: декодер faster-whisper не совместим с новым PyAV
    segs, _ = asr.transcribe(y16, language='en', word_timestamps=True, vad_filter=True, beam_size=5,
                             condition_on_previous_text=False)
    out = [dict(start=s.start, end=s.end, text=s.text,
                words=[dict(start=w.start, end=w.end, word=w.word) for w in (s.words or [])]) for s in segs]
    json.dump(out, open(cache, 'w'))
    return out


def phrases(segs):
    """сегменты whisper -> фразы 1.5-11 с"""
    units = []
    for s in segs:
        ws = [w for w in s['words'] if w['end'] > w['start']]
        if not ws:
            continue
        # длинный сегмент режем на самой большой паузе между словами, пока не влезет в MAX_DUR
        stack = [ws]
        while stack:
            cur = stack.pop()
            if cur[-1]['end'] - cur[0]['start'] <= MAX_DUR or len(cur) < 2:
                units.append(cur)
                continue
            gaps = [cur[i + 1]['start'] - cur[i]['end'] for i in range(len(cur) - 1)]
            k = int(np.argmax(gaps)) + 1
            stack += [cur[k:], cur[:k]]
    units.sort(key=lambda u: u[0]['start'])
    # склейка соседних коротких единиц, если пауза маленькая и итог не длиннее MAX_DUR
    out = []
    for u in units:
        if out and u[0]['start'] - out[-1][-1]['end'] < MAX_GAP and u[-1]['end'] - out[-1][0]['start'] <= MAX_DUR \
                and out[-1][-1]['end'] - out[-1][0]['start'] < 4.0:
            out[-1] = out[-1] + u
        else:
            out.append(u)
    return [u for u in out if MIN_DUR <= u[-1]['end'] - u[0]['start'] <= MAX_DUR]


def norm_words(text):
    return re.sub(r"[^a-z' ]", ' ', text.lower()).split()


def ngrams(words, n):
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def main():
    from faster_whisper import WhisperModel
    from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

    asr = WhisperModel('large-v3', device='mps' if DEVICE == 'mps' else DEVICE, compute_type='float16')
    fe = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base-sv')
    sv = WavLMForXVector.from_pretrained('microsoft/wavlm-base-sv').eval().to(DEVICE)

    def embed(y, sr):
        y16 = librosa.resample(y, orig_sr=sr, target_sr=16000)
        with torch.no_grad():
            e = sv(**fe(y16, sampling_rate=16000, return_tensors='pt').to(DEVICE)).embeddings
        return torch.nn.functional.normalize(e, dim=-1)[0].cpu().numpy()

    files = sorted(f for f in os.listdir(RAW) if f.endswith('.wav'))
    if a.order:
        # приоритет по префиксам имён: например film_dos,film_bofa -> сначала фильмы, потом нарезки
        pri = a.order.split(',')
        rank = lambda f: next((i for i, pfx in enumerate(pri) if f.startswith(pfx)), len(pri))
        files = sorted(files, key=lambda f: (rank(f), f))
    print('порядок исходников:', files, flush=True)

    for f in os.listdir(os.path.join(OUT, 'wavs')):
        os.remove(os.path.join(OUT, 'wavs', f))
    rows, seen3, seen2 = [], set(), set()
    for f in files:
        vocals = separate(os.path.join(RAW, f))
        segs = transcribe(asr, vocals, f)
        y, _ = librosa.load(vocals, sr=SR, mono=True)
        for u in phrases(segs):
            a0, b0 = max(0.0, u[0]['start'] - 0.1), u[-1]['end'] + 0.15
            text = ''.join(w['word'] for w in u).strip()
            seg = y[int(a0 * SR):int(b0 * SR)]
            if len(text) / (b0 - a0) < MIN_CPS or np.abs(seg).max() < 1e-3:
                continue
            words = norm_words(text)
            g3, g2 = ngrams(words, 3), ngrams(words, 2)
            if g3:
                dup = len(g3 & seen3) / len(g3) >= 0.5
            else:
                dup = bool(g2) and len(g2 & seen2) / len(g2) >= 0.5
            seg = seg / np.abs(seg).max() * 0.9
            sid = f'{os.path.splitext(f)[0]}_{int(a0 * 100):07d}'
            sf.write(os.path.join(OUT, 'wavs', sid + '.wav'), seg, SR)
            rows.append(dict(id=sid, source=f, start=round(a0, 2), end=round(b0, 2), dur=round(b0 - a0, 2),
                             text=text, dup=dup, emb=embed(seg, SR)))
        # n-граммы всей расшифровки исходника, а не только оставленных фраз: иначе пропускаются дубли,
        # которые в другом исходнике порезаны по-другому
        all_words = norm_words(' '.join(s['text'] for s in segs))
        seen3 |= ngrams(all_words, 3)
        seen2 |= ngrams(all_words, 2)
        n_f = sum(r['source'] == f for r in rows)
        n_new = sum(r['source'] == f and not r['dup'] for r in rows)
        print(f'{f}: фраз {n_f}, новых {n_new}', flush=True)

    df = pd.DataFrame(rows)
    E = np.stack(df.emb.values)
    if a.seed:
        seeds = [os.path.join(a.seed, x) for x in sorted(os.listdir(a.seed)) if x.endswith('.wav')]
        c = np.mean([embed(librosa.load(s, sr=SR, mono=True)[0], SR) for s in seeds], 0)
        print('центроид по эталонам:', len(seeds), flush=True)
    else:
        c = np.median(E, 0)
        c /= np.linalg.norm(c)
        sim = E @ c
        c = E[sim >= np.quantile(sim, 0.5)].mean(0)
    c /= np.linalg.norm(c)
    df['spk_sim'] = E @ c
    df.drop(columns='emb').to_csv(os.path.join(OUT, 'segments.csv'), index=False)
    np.save(os.path.join(OUT, 'emb.npy'), E)
    u = df[~df.dup]
    print(json.dumps(dict(n=len(df), minutes=round(df.dur.sum() / 60, 2), dup=int(df.dup.sum()),
                          uniq_minutes=round(u.dur.sum() / 60, 2),
                          uniq_min_by_sim={t: round(u[u.spk_sim >= t].dur.sum() / 60, 2) for t in (0.7, 0.8, 0.85, 0.9)})))


if __name__ == '__main__':
    main()
