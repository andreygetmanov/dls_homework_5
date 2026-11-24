"""segments.csv -> датасет LJSpeech с разбиением.

    .venv/bin/python scripts/make_splits.py smaug --min_sim 0.6 --test dos_0827802,dos_1031591,bofa_0061654

- выкидываются дубли, фразы короче 1.5 с, фразы с низкой близостью к эталонам спикера (--min_sim)
  и с расхождением повторной расшифровки и текста (WER >= --max_wer, если в segments.csv есть столбец wer)
- test: 3 референса, заданные явно (--test), чтобы не менялись при пересборке; копируются в data/test/<name>.
  Без --test берутся 3 фразы по 5-10 с из разных мест записи с самой высокой близостью к спикеру. В обучение не идут
- eval: ~10% оставшихся фраз (не меньше 4), по ним выбирается лучший чекпойнт
- full: всё остальное, 1min: случайные фразы из full суммарно ~60 с
Выход: data/ljs/<name>/wavs/*.wav, metadata_{full,1min,eval}.csv (id|text|text), test_refs.txt, split.csv
"""
import argparse
import os
import shutil

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = argparse.ArgumentParser()
p.add_argument('name')
p.add_argument('--min_sim', type=float, default=0.7)
p.add_argument('--max_wer', type=float, default=0.5)
p.add_argument('--test', default=None, help='id трёх тестовых референсов через запятую')
p.add_argument('--seed', type=int, default=0)
a = p.parse_args()

SEG = os.path.join(ROOT, 'data', 'segments', a.name)
OUT = os.path.join(ROOT, 'data', 'ljs', a.name)
rng = np.random.default_rng(a.seed)

df = pd.read_csv(os.path.join(SEG, 'segments.csv'))
df['text'] = df.text.str.replace('|', ' ', regex=False).str.strip()
keep = ~df.dup & (df.spk_sim >= a.min_sim) & (df.dur >= 1.5)
if 'wer' in df:
    keep &= df.wer < a.max_wer
if 'drop' in df:  # ручная проверка (Картман)
    keep &= ~df['drop'].astype(bool)
ok = df[keep].copy()
print(f'всего фраз {len(df)}, после фильтра {len(ok)} ({ok.dur.sum() / 60:.2f} мин)')

if a.test:
    ids = a.test.split(',')
    test = [ok.index[ok.id == t][0] for t in ids]
else:
    # кандидаты 5-10 с, берём лучших по близости, но не ближе 30 с друг к другу по исходнику
    cand = ok[(ok.dur >= 5) & (ok.dur <= 10)].sort_values('spk_sim', ascending=False)
    test = []
    for i, r in cand.iterrows():
        if all(r.source != ok.loc[j, 'source'] or abs(r.start - ok.loc[j, 'start']) > 30 for j in test):
            test.append(i)
        if len(test) == 3:
            break
assert len(test) == 3, 'не нашлось трёх референсов 5-10 с'
rest = ok.drop(index=test)

n_eval = max(4, int(round(0.1 * len(rest))))
eval_idx = rng.choice(rest.index, n_eval, replace=False)
train = rest.drop(index=eval_idx)

perm = rng.permutation(train.index)
one, total = [], 0.0
for i in perm:
    if total >= 60:
        break
    one.append(i)
    total += train.loc[i, 'dur']

ok['split'] = 'full'
ok.loc[test, 'split'] = 'test'
ok.loc[eval_idx, 'split'] = 'eval'
ok['in_1min'] = ok.index.isin(one)

if os.path.exists(OUT):
    shutil.rmtree(OUT)
os.makedirs(os.path.join(OUT, 'wavs'))
for i, r in ok.iterrows():
    shutil.copy(os.path.join(SEG, 'wavs', r.id + '.wav'), os.path.join(OUT, 'wavs', r.id + '.wav'))


def write_meta(rows, fname):
    with open(os.path.join(OUT, fname), 'w', encoding='utf-8') as f:
        for _, r in rows.iterrows():
            f.write(f'{r.id}|{r.text}|{r.text}\n')


write_meta(ok[ok.split == 'full'], 'metadata_full.csv')
write_meta(ok[ok.in_1min], 'metadata_1min.csv')
write_meta(ok[ok.split == 'eval'], 'metadata_eval.csv')
with open(os.path.join(OUT, 'test_refs.txt'), 'w') as f:
    f.write('\n'.join(f'wavs/{ok.loc[i, "id"]}.wav' for i in test) + '\n')
ok.to_csv(os.path.join(OUT, 'split.csv'), index=False)
TEST = os.path.join(ROOT, 'data', 'test', a.name)
os.makedirs(TEST, exist_ok=True)
for i in test:
    shutil.copy(os.path.join(SEG, 'wavs', ok.loc[i, 'id'] + '.wav'), TEST)
ok.loc[test].to_csv(os.path.join(TEST, 'refs.csv'), index=False)

for name, rows in [('full', ok[ok.split == 'full']), ('1min', ok[ok.in_1min]), ('eval', ok[ok.split == 'eval']),
                   ('test', ok[ok.split == 'test'])]:
    print(f'{name:5s}: {len(rows):4d} фраз, {rows.dur.sum():7.1f} с, средняя {rows.dur.mean():.2f} с')
