"""Собирает ноутбук с решением из шаблона hw.ipynb и файлов scripts/cells/*.py|*.md и выполняет его.

    .venv/bin/python scripts/build_nb.py           # собрать и выполнить -> hw_Andrey_Getmanov.ipynb
    .venv/bin/python scripts/build_nb.py --no-run  # только собрать

Ячейки шаблона сохраняются; ячейки с «ВАШ КОД» заменяются решением, ниже текстовых добавляются пояснения.
Обучение и генерация трёх моделей шли на Kaggle (kaggle/code/run.py), в ноутбуке читаются их логи и wav
"""
import copy
import json
import os
import sys

S = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(S)
tpl = json.load(open(os.path.join(ROOT, 'hw.ipynb')))
orig = tpl['cells']


def src_lines(text):
    lines = text.rstrip('\n').split('\n')
    return [l + '\n' for l in lines[:-1]] + [lines[-1]]


def code(name):
    text = open(os.path.join(S, 'cells', name + '.py')).read()
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': src_lines(text)}


def md(name):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': src_lines(open(os.path.join(S, 'cells', name + '.md')).read())}


def o(i):
    return copy.deepcopy(orig[i])


cells = [
    o(0), o(1), md('c01_env'), code('c01_versions'),
    o(2), md('c02_voice'), code('c03_prepare'), code('c04_describe'),
    o(5), md('c05_answer'), code('c06_train_1min'), code('c07_train_full'), md('c07_note'),
    o(8), code('c09_generate'), code('c10_listen'),
    o(11), code('c12_similarity'), code('c13_print'),
    md('c14_extra'), code('c14_extra'), md('c15_conclusions'),
    # дополнительный раздел: Картман (MTV), kaggle/cartman/code/run.py
    md('x01_cartman'), code('x02_cartman_data'), md('x03_cartman_baseline'), code('x03_cartman_baseline'),
    md('x04_cartman_zs'), code('x04_cartman_zs'), md('x04_cartman_zs_verdict'), code('x05_cartman_train'),
    code('x06_cartman_compare'), code('x06_cartman_listen'), md('x07_cartman_conclusions'),
]
nb = {k: copy.deepcopy(v) for k, v in tpl.items() if k != 'cells'}
nb['cells'] = cells
out = os.path.join(ROOT, 'hw_Andrey_Getmanov.ipynb')
if '--no-run' not in sys.argv:
    import nbformat
    from nbclient import NotebookClient
    nbo = nbformat.reads(json.dumps(nb), as_version=4)
    NotebookClient(nbo, timeout=1800, kernel_name='python3', resources={'metadata': {'path': ROOT}}).execute()
    nb = nbo
json.dump(nb, open(out, 'w'), ensure_ascii=False, indent=1)
print('wrote', out, len(cells), 'cells')
