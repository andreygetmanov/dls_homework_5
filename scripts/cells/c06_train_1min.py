# 3.3. Дообучите модель на 1 минуте данных
# запуск на Kaggle T4 (kaggle/code/run.py); параметры из рецепта: lr 5e-6, AdamW (0.9, 0.96), wd 1e-2, batch 4,
# условие 3-6 с из той же фразы; eval loss на 6 других фразах каждые 10 шагов, сохраняется лучший по eval loss_mel_ce
#   python scripts/train_xtts.py --data data/ljs/smaug --meta metadata_1min.csv --out runs/1min --steps 300 --eval_every 10
def summary(name):
    log = pd.read_csv(f'outputs/kaggle/runs/{name}/log.csv')
    ev = log[log.split == 'eval']
    best = ev.loc[ev.loss_mel_ce.idxmin()]
    print(f'{name}: {int(log.step.max())} шагов за {log.seconds.max() / 60:.1f} мин; eval loss_mel_ce: '
          f'исходная модель {ev.loss_mel_ce.iloc[0]:.3f}, лучший {best.loss_mel_ce:.3f} на шаге {int(best.step)}, '
          f'в конце {ev.loss_mel_ce.iloc[-1]:.3f}')
    return log

log_1min = summary('1min')
