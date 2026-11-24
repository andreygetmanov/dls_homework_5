"""Kaggle (GPU), дополнительный раздел: Картман (MTV, русский). То же, что kaggle/code/run.py, но language=ru.

Вход: датасет dls-hw5-cartman (ljs/cartman: wavs, metadata_*.csv, test_refs.txt; scripts/*.py)
Выход в /kaggle/working: runs/<1min|full>/log.csv, gens/<zs|1min|full>/*.wav, versions.txt
"""
import glob
import os
import subprocess
import sys
import time

W = '/kaggle/working'
IN = os.path.dirname(glob.glob('/kaggle/input/**/test_refs.txt', recursive=True)[0])  # .../ljs/cartman
ROOT_IN = os.path.dirname(os.path.dirname(IN))
SCRIPTS = os.path.join(ROOT_IN, 'scripts')
os.environ['COQUI_TOS_AGREED'] = '1'
os.environ['HF_HUB_DISABLE_XET'] = '1'


def sh(cmd):
    print('$', cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


sh('pip install -q coqui-tts==0.28.0 transformers==4.57.6 2>&1 | tail -3')
import torch, torchaudio  # noqa: E402
try:
    torchaudio.load(glob.glob(os.path.join(IN, 'wavs', '*.wav'))[0])
except Exception as e:  # torchaudio >= 2.9 читает только через torchcodec
    print('torchaudio.load:', repr(e)[:200], flush=True)
    sh('pip install -q torchcodec 2>&1 | tail -3')
sh(f'{sys.executable} -c "import torch, torchaudio, transformers, TTS; '
   f'print(torch.__version__, torchaudio.__version__, transformers.__version__, TTS.__version__, '
   f'torch.cuda.get_device_name(0))" | tee {W}/versions.txt')

FILES = f'{W}/xtts_files'
STEPS = {'1min': (300, 10), 'full': (600, 25)}
for name, meta in (('1min', 'metadata_1min.csv'), ('full', 'metadata_full.csv')):
    steps, ev = STEPS[name]
    t = time.time()
    sh(f'{sys.executable} {SCRIPTS}/train_xtts.py --data {IN} --meta {meta} --out {W}/runs/{name} '
       f'--steps {steps} --eval_every {ev} --lang ru --files {FILES} 2>&1 | grep -v -i warn')
    print(f'{name}: обучение {time.time() - t:.0f} с', flush=True)

sys.path.insert(0, SCRIPTS)
import xtts_eval as xe  # noqa: E402

ref0 = os.path.join(IN, open(os.path.join(IN, 'test_refs.txt')).readline().strip())
for name, ckpt in (('zs', None), ('1min', f'{W}/runs/1min/best_model.pth'), ('full', f'{W}/runs/full/best_model.pth')):
    t = time.time()
    model = xe.load_xtts(ckpt, base=FILES)
    xe.synthesize(model, ref0, xe.SENTENCES_RU, f'{W}/gens/{name}', name, lang='ru')
    print(f'{name}: генерация {time.time() - t:.0f} с', flush=True)
    del model
    torch.cuda.empty_cache()

for f in glob.glob(f'{W}/runs/*/best_model.pth'):
    os.remove(f)  # 2 ГБ каждый, локально не нужны
sh(f'rm -rf {FILES}')
print('DONE', flush=True)
