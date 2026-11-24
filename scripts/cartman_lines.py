"""Реплики Картмана из русской MTV-озвучки «Южного парка» (сезон 1, 8 серий).

    .venv/bin/python scripts/cartman_lines.py [--align_only]

Меток говорящего в субтитрах нет, а в озвучке голоса похожи, поэтому кто говорит, берётся из сценариев
(southpark.fandom.com, data/scripts/cartman/S01E0X.txt, строки {{ScriptDialog|говорящий|текст}}):
1. слова сценария (с говорящим) выравниваются со словами английских субтитров (difflib); строка субтитров
   считается репликой Картмана, если >= 80% её совпавших слов из реплик Картмана и совпало >= 50% слов
2. подряд идущие строки Картмана (пауза < MAX_GAP) склеиваются в фразы <= MAX_DUR; окно в дорожке MTV:
   начало - 0.3 с, конец + 0.6 с (дубляж не совпадает с оригиналом до кадра), но не заходя на чужие строки
3. вокал из demucs, тишина по краям обрезается, текст: whisper large-v3 (ru) по самому фрагменту
4. WavLM-близость к центроиду; центроид итеративно: медиана по всем кандидатам, затем среднее по ближайшей половине.
   Для сравнения тем же способом режутся строки других персонажей (others.csv): база «чужой голос той же озвучки»
5. ручная проверка (data/scripts/cartman/review.csv: id,action,text): каждая пара «английский текст / расшифровка
   дубляжа» сверена по смыслу; drop, если в фрагмент попала чужая реплика, перевод не совпадает с репликой или
   whisper ошибся; fix — исправленный текст. Применяется без пересчёта: --review
Результат: data/segments/cartman/wavs/*.wav (22050 Гц), segments.csv в формате prep_data.py
"""
import argparse
import difflib
import json
import os
import re

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, 'data', 'scripts', 'cartman')
SRT = os.path.join(ROOT, 'data', 'srt', 'cartman')
SEP = os.path.join(ROOT, 'data', 'separated', 'cartman', 'htdemucs')
OUT = os.path.join(ROOT, 'data', 'segments', 'cartman')
SR = 22050
MAX_GAP, MAX_DUR = 0.8, 11.0
EPISODES = [f'S01E0{i}' for i in range(1, 9)]
DEVICE = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
p = argparse.ArgumentParser()
p.add_argument('--align_only', action='store_true')
p.add_argument('--review', action='store_true', help='только применить review.csv к segments.csv')
p.add_argument('--episodes', default=','.join(EPISODES), help='для проверки на части серий')
args = p.parse_args()
EPISODES = args.episodes.split(',')


def words(text):
    return re.sub(r"[^a-z0-9' ]", ' ', text.lower().replace('*', '')).split()


def read_script(path):
    """[(слово, говорящий)]; ремарки в [скобках] и (бормотание Кенни) выкидываются"""
    out = []
    for m in re.finditer(r'\{\{ScriptDialog\|([^|}]*)\|(.*?)\}\}\s*$', open(path).read(), flags=re.M):
        spk, text = m.group(1).strip(), m.group(2)
        text = re.sub(r"''\[.*?\]''|\[.*?\]", ' ', text)
        text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]*)\]\]', r'\1', text)
        text = re.sub(r'<[^>]+>|\(.*?\)', ' ', text).replace("''", '')
        is_c = 'cartman' in spk.lower() and 'liane' not in spk.lower()
        name = spk.lower().split()
        if len(name) > 1 and name[0] not in ('mr.', 'mrs.', 'ms.', 'officer', 'big', 'the', 'uncle', 'dr.'):
            name = name[:1]  # «Kyle Broflovski» и «Kyle» -> kyle
        who = 'cartman' if is_c and ',' not in spk else ('mixed' if is_c else ' '.join(name))
        out += [(w, who) for w in words(text)]
    return out


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
        raw = ' '.join(L[2:])
        text = re.sub(r'<[^>]+>|\[.*?\]|\(.*?\)', ' ', raw)
        items.append(dict(i=int(L[0]), a=t(a), c=t(c), text=re.sub(r'\s+', ' ', text).strip(),
                          two=bool(re.search(r'(^|\s)-\s', raw.strip()[1:])), song='♪' in raw))
    return items


def label(ep):
    sw = read_script(os.path.join(SCRIPTS, ep + '.txt'))
    cues = read_srt(os.path.join(SRT, ep + '_en.srt'))
    cw = [(w, k) for k, c in enumerate(cues) for w in words(c['text'])]
    sm = difflib.SequenceMatcher(None, [w for w, _ in sw], [w for w, _ in cw], autojunk=False)
    spk = [[] for _ in cues]
    for blk in sm.get_matching_blocks():
        for d in range(blk.size):
            spk[cw[blk.b + d][1]].append(sw[blk.a + d][1])
    for c, s in zip(cues, spk):
        n = len(words(c['text']))
        c['n'], c['matched'] = n, len(s)
        c['who'] = max(set(s), key=s.count) if s else None
        c['purity'] = s.count(c['who']) / len(s) if s else 0
        c['cartman'] = (c['who'] == 'cartman' and c['purity'] >= 0.8 and len(s) >= 0.5 * n and not c['two']
                        and not c['song'])
    return cues


def groups_of(cues, pick):
    """подряд идущие строки одного говорящего -> фразы <= MAX_DUR"""
    runs = []
    for k, c in enumerate(cues):
        if not pick(c):
            continue
        if runs and runs[-1][-1] == k - 1 and c['a'] - cues[k - 1]['c'] < MAX_GAP \
                and c['c'] - cues[runs[-1][0]]['a'] <= MAX_DUR:
            runs[-1].append(k)
        else:
            runs.append([k])
    return runs


def window(cues, run):
    a, b = cues[run[0]]['a'] - 0.3, cues[run[-1]]['c'] + 0.6
    prev_end = cues[run[0] - 1]['c'] if run[0] > 0 else -1
    next_start = cues[run[-1] + 1]['a'] if run[-1] + 1 < len(cues) else 1e9
    return max(a, prev_end + 0.05), min(b, next_start - 0.05)


HALLUC = re.compile(r'субтитр|редактор|корректор|продолжение следует|подпис|спасибо за просмотр', re.I)


def apply_review():
    df = pd.read_csv(os.path.join(OUT, 'segments.csv'))
    rv = pd.read_csv(os.path.join(SCRIPTS, 'review.csv'), dtype=str).fillna('')
    df['text'] = df.asr
    df['drop'] = df.asr.fillna('').str.contains(HALLUC)
    cps = df.asr.fillna('').str.len() / df.dur
    df['drop'] |= (cps < 4) | (cps > 25)
    for r in rv.itertuples():
        m = df.id == r.id
        assert m.sum() == 1, r.id
        if r.action == 'drop':
            df.loc[m, 'drop'] = True
        elif r.action == 'fix':
            df.loc[m, 'text'] = r.text
    df.to_csv(os.path.join(OUT, 'segments.csv'), index=False)
    k = df[~df['drop']]
    print(f'после проверки {len(k)} из {len(df)} фраз, {k.dur.sum() / 60:.2f} мин', flush=True)


def main():
    if args.review:
        return apply_review()
    stats, cands, others = {}, [], []
    for ep in EPISODES:
        cues = label(ep)
        stats[ep] = dict(cues=len(cues), cartman=sum(c['cartman'] for c in cues),
                         matched=round(np.mean([c['matched'] / max(1, c['n']) for c in cues]), 3))
        for run in groups_of(cues, lambda c: c['cartman']):
            a, b = window(cues, run)
            cands.append((ep, a, b, ' '.join(cues[k]['text'] for k in run), ','.join(str(cues[k]['i']) for k in run)))
        # чужие голоса для базы: одиночные строки других персонажей с той же чистотой разметки
        for k, c in enumerate(cues):
            if c['who'] not in (None, 'cartman', 'mixed') and c['purity'] >= 0.8 and c['matched'] >= 0.5 * c['n'] \
                    and not c['two'] and not c['song'] and c['c'] - c['a'] >= 1.5:
                a, b = window(cues, [k])
                others.append((ep, a, b, c['text'], c['who']))
        pd.DataFrame(cues).to_csv(os.path.join(SCRIPTS, ep + '_labels.csv'), index=False)
    print(json.dumps(stats), flush=True)
    print('кандидатов', len(cands), 'суммарно', round(sum(b - a for _, a, b, *_ in cands) / 60, 2), 'мин', flush=True)
    if args.align_only:
        return

    from faster_whisper import WhisperModel
    from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
    asr = WhisperModel('large-v3', device=DEVICE, compute_type='float16')
    fe = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base-sv')
    sv = WavLMForXVector.from_pretrained('microsoft/wavlm-base-sv').eval().to(DEVICE)

    def embed(x):
        with torch.no_grad():
            e = sv(**fe(librosa.resample(x, orig_sr=SR, target_sr=16000), sampling_rate=16000,
                         return_tensors='pt').to(DEVICE)).embeddings
        return torch.nn.functional.normalize(e, dim=-1)[0].cpu().numpy()

    def cut(y, a, b):
        seg = y[int(a * SR):int(b * SR)]
        if len(seg) < SR * 0.5 or np.abs(seg).max() < 1e-3:
            return None, a, b
        _, (i0, i1) = librosa.effects.trim(seg, top_db=35, frame_length=1024, hop_length=256)
        i0, i1 = max(0, i0 - int(0.1 * SR)), min(len(seg), i1 + int(0.15 * SR))
        return seg[i0:i1], a + i0 / SR, a + i1 / SR

    def transcribe(seg):
        segs, _ = asr.transcribe(librosa.resample(seg, orig_sr=SR, target_sr=16000), language='ru', beam_size=5,
                                 vad_filter=False, condition_on_previous_text=False)
        segs = list(segs)
        text = ' '.join(s.text.strip() for s in segs).strip()
        lp = float(np.mean([s.avg_logprob for s in segs])) if segs else -9.0
        nsp = float(np.max([s.no_speech_prob for s in segs])) if segs else 1.0
        return text, lp, nsp

    os.makedirs(os.path.join(OUT, 'wavs'), exist_ok=True)
    for f in os.listdir(os.path.join(OUT, 'wavs')):
        os.remove(os.path.join(OUT, 'wavs', f))
    rows, E, orows, OE = [], [], [], []
    for ep in EPISODES:
        y, _ = librosa.load(os.path.join(SEP, ep + '_mtv', 'vocals.wav'), sr=SR, mono=True)
        for e, a, b, en, srt in [c for c in cands if c[0] == ep]:
            seg, a, b = cut(y, a, b)
            if seg is None or b - a < 1.0:
                continue
            text, lp, nsp = transcribe(seg)
            seg = seg / np.abs(seg).max() * 0.9
            sid = f'{ep}_{int(a * 100):06d}'
            sf.write(os.path.join(OUT, 'wavs', sid + '.wav'), seg, SR)
            E.append(embed(seg))
            rows.append(dict(id=sid, source=ep, start=round(a, 2), end=round(b, 2), dur=round(b - a, 2), text=text,
                             asr=text, dup=False, srt=srt, en=en, logprob=round(lp, 3), no_speech=round(nsp, 3)))
            print(f"{sid} {b - a:5.2f}s lp={lp:.2f} | {en} | {text}", flush=True)
        for e, a, b, en, who in [c for c in others if c[0] == ep]:
            seg, a, b = cut(y, a, b)
            if seg is None or b - a < 1.0:
                continue
            OE.append(embed(seg / np.abs(seg).max() * 0.9))
            orows.append(dict(source=ep, start=round(a, 2), end=round(b, 2), who=who, en=en))
    E, OE = np.stack(E), np.stack(OE)
    c = np.median(E, 0)
    c /= np.linalg.norm(c)
    for _ in range(3):
        s = E @ c
        c = E[s >= np.quantile(s, 0.5)].mean(0)
        c /= np.linalg.norm(c)
    df = pd.DataFrame(rows)
    df['spk_sim'] = E @ c
    df.to_csv(os.path.join(OUT, 'segments.csv'), index=False)
    np.save(os.path.join(OUT, 'emb.npy'), E)
    od = pd.DataFrame(orows)
    od['spk_sim'] = OE @ c
    od.to_csv(os.path.join(OUT, 'others.csv'), index=False)
    np.save(os.path.join(OUT, 'emb_others.npy'), OE)
    np.save(os.path.join(OUT, 'centroid.npy'), c)
    q = lambda v: np.round(np.quantile(v, [0.1, 0.25, 0.5, 0.75, 0.9]), 3).tolist()
    print(json.dumps(dict(n=len(df), minutes=round(df.dur.sum() / 60, 2), cartman_sim_q=q(df.spk_sim),
                          others_sim_q=q(od.spk_sim),
                          others_by_who=od.groupby('who').spk_sim.agg(['count', 'mean']).round(3)
                          .sort_values('count', ascending=False).head(8).to_dict('index'),
                          minutes_by_sim={t: round(df[df.spk_sim >= t].dur.sum() / 60, 2)
                                          for t in (0.6, 0.7, 0.8, 0.85, 0.9)})), flush=True)


if __name__ == '__main__':
    main()
