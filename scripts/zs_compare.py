"""Zero-shot XTTS-v2 для кандидатов: 10 предложений по первому тестовому референсу,
similarity референсов друг с другом (GT) и всех трёх референсов со всеми генерациями.

    .venv/bin/python scripts/zs_compare.py smaug meryl
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xtts_eval as xe

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
names = sys.argv[1:]

model = xe.load_xtts()
sv = xe.SpeakerSim()
print('device:', next(model.parameters()).device, flush=True)

res = {}
for n in names:
    d = os.path.join(ROOT, 'data', 'ljs', n)
    refs = [os.path.join(d, l.strip()) for l in open(os.path.join(d, 'test_refs.txt')) if l.strip()]
    t = time.time()
    gens = xe.synthesize(model, refs[0], xe.SENTENCES, os.path.join(ROOT, 'outputs', 'zs_compare', n), 'zs')
    gen_time = time.time() - t
    gt = sv.refs_pairwise(refs)
    M = sv.matrix(refs, gens)
    res[n] = dict(gt_pairs=gt.round(4).tolist(), gt_mean=float(gt.mean()),
                  zs_mean=float(M.mean()), zs_std=float(M.std()),
                  zs_ref0_mean=float(M[0].mean()), zs_other_refs_mean=float(M[1:].mean()),
                  zs_per_ref=M.mean(1).round(4).tolist(), gen_seconds=round(gen_time, 1))
    print(n, json.dumps(res[n], ensure_ascii=False), flush=True)

os.makedirs(os.path.join(ROOT, 'outputs'), exist_ok=True)
json.dump(res, open(os.path.join(ROOT, 'outputs', 'zs_compare.json'), 'w'), indent=1)
