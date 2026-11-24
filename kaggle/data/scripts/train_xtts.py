"""Дообучение GPT-части XTTS-v2 по рецепту recipes/ljspeech/xtts_v2/train_gpt_xtts.py из coqui-tts.

    python scripts/train_xtts.py --data data/ljs/smaug --meta metadata_1min.csv --out runs/1min --steps 300

Модель, датасет, лосс и оптимизатор берутся из GPTTrainer как в рецепте (lr 5e-6, AdamW betas 0.9/0.96, wd 1e-2
только на веса, оптимизируется только xtts.gpt; DVAE и HiFi-GAN заморожены). Отличие одно: цикл обучения свой,
потому что Trainer из coqui умеет только CUDA, а здесь обучение идёт и на MPS (Apple), и на CUDA (Kaggle).
Каждые --eval_every шагов считается loss на metadata_eval.csv (другие фразы того же спикера);
best_model.pth сохраняется по минимуму eval loss_mel_ce. Лог: <out>/log.csv
"""
import argparse
import csv
import os
import random
import time

import numpy as np
import torch

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.xtts_config import XttsAudioConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig
from TTS.utils.manage import ModelManager

p = argparse.ArgumentParser()
p.add_argument('--data', required=True)
p.add_argument('--meta', required=True)
p.add_argument('--eval_meta', default='metadata_eval.csv')
p.add_argument('--out', required=True)
p.add_argument('--steps', type=int, default=300)
p.add_argument('--eval_every', type=int, default=10)
p.add_argument('--batch', type=int, default=4)
p.add_argument('--lr', type=float, default=5e-6)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--files', default='xtts_files', help='исходные файлы XTTS-v2 (model.pth, dvae.pth, ...)')
a = p.parse_args()

DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
random.seed(a.seed)
np.random.seed(a.seed)
torch.manual_seed(a.seed)
os.makedirs(a.out, exist_ok=True)

os.makedirs(a.files, exist_ok=True)
HF = 'https://huggingface.co/coqui/XTTS-v2/resolve/main/'
names = ['dvae.pth', 'mel_stats.pth', 'vocab.json', 'model.pth', 'config.json']
missing = [n for n in names if not os.path.isfile(os.path.join(a.files, n))]
if missing:
    ModelManager._download_model_files([HF + n for n in missing], a.files, progress_bar=False)
F = {n: os.path.join(a.files, n) for n in names}

dataset = BaseDatasetConfig(formatter='ljspeech', dataset_name='ft', path=a.data,
                            meta_file_train=a.meta, meta_file_val=a.eval_meta, language='en')
model_args = GPTArgs(
    max_conditioning_length=132300,  # 6 с
    min_conditioning_length=66150,   # 3 с
    debug_loading_failures=False,
    max_wav_length=255995,           # ~11.6 с
    max_text_length=200,
    mel_norm_file=F['mel_stats.pth'],
    dvae_checkpoint=F['dvae.pth'],
    xtts_checkpoint=F['model.pth'],
    tokenizer_file=F['vocab.json'],
    gpt_num_audio_tokens=1026,
    gpt_start_audio_token=1024,
    gpt_stop_audio_token=1025,
    gpt_use_masking_gt_prompt_approach=True,
    gpt_use_perceiver_resampler=True,
)
config = GPTTrainerConfig(
    output_path=a.out, model_args=model_args,
    audio=XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000),
    batch_size=a.batch, eval_batch_size=a.batch, num_loader_workers=0, num_eval_loader_workers=0,
    optimizer='AdamW', optimizer_wd_only_on_weights=True,
    optimizer_params={'betas': [0.9, 0.96], 'eps': 1e-8, 'weight_decay': 1e-2},
    lr=a.lr, training_seed=a.seed, test_sentences=[], datasets=[dataset],
)

model = GPTTrainer(config)
model.to(DEVICE)
train_samples, eval_samples = load_tts_samples(config, eval_split=True)
train_loader = model.get_data_loader(config, is_eval=False, samples=train_samples, verbose=False)
eval_loader = model.get_data_loader(config, is_eval=True, samples=eval_samples, verbose=False)
optimizer = model.get_optimizer()
n_trainable = sum(p.numel() for g in optimizer.param_groups for p in g['params'])
print(f'device {DEVICE}; train {len(train_samples)} фраз, eval {len(eval_loader.dataset.samples)}; '
      f'обучаемых параметров {n_trainable / 1e6:.1f}M из {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M',
      flush=True)


def to_device(batch):
    batch = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in batch.items()}
    return model.format_batch_on_device(batch)


@torch.no_grad()
def evaluate():
    model.eval()
    tot = {}
    n = 0
    for batch in eval_loader:
        _, ld = model.eval_step(to_device(batch), None)
        bs = batch['wav_lengths'].shape[0]
        for k, v in ld.items():
            tot[k] = tot.get(k, 0.0) + float(v) * bs
        n += bs
    return {k: v / n for k, v in tot.items()}


def train_mode():
    model.eval()          # DVAE, mel-спектрограммы и т.д. в eval, как в GPTTrainer.on_train_epoch_start
    model.xtts.gpt.train()


log = open(os.path.join(a.out, 'log.csv'), 'w', newline='')
w = csv.writer(log)
w.writerow(['step', 'split', 'loss', 'loss_mel_ce', 'loss_text_ce', 'seconds'])
best, step, t0 = float('inf'), 0, time.time()


def do_eval():
    global best
    ev = evaluate()
    w.writerow([step, 'eval', ev['loss'], ev['loss_mel_ce'], ev['loss_text_ce'], round(time.time() - t0, 1)])
    log.flush()
    mark = ''
    if ev['loss_mel_ce'] < best:
        best = ev['loss_mel_ce']
        torch.save({'model': model.state_dict(), 'step': step, 'eval': ev}, os.path.join(a.out, 'best_model.pth'))
        mark = ' *'
    print(f'step {step:4d} eval loss_mel_ce {ev["loss_mel_ce"]:.4f} text {ev["loss_text_ce"]:.4f}{mark}', flush=True)


do_eval()  # исходная модель
train_mode()
run = []
while step < a.steps:
    for batch in train_loader:
        batch = to_device(batch)
        _, ld = model.train_step(batch, None)
        optimizer.zero_grad(set_to_none=True)
        ld['loss'].backward()
        optimizer.step()
        step += 1
        run.append([float(ld[k]) for k in ('loss', 'loss_mel_ce', 'loss_text_ce')])
        w.writerow([step, 'train', *run[-1], round(time.time() - t0, 1)])
        if step % a.eval_every == 0 or step == a.steps:
            m = np.mean(run, 0)
            print(f'step {step:4d} train loss_mel_ce {m[1]:.4f} ({(time.time() - t0) / step:.2f} с/шаг)', flush=True)
            run = []
            do_eval()
            train_mode()
        if step >= a.steps:
            break
log.close()
print('best eval loss_mel_ce', round(best, 4), '->', os.path.join(a.out, 'best_model.pth'), flush=True)
