# 3.4. Дообучите модель на всех данных
#   python scripts/train_xtts.py --data data/ljs/smaug --meta metadata_full.csv --out runs/full --steps 600 --eval_every 25
import matplotlib.pyplot as plt

log_full = summary('full')
fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
for ax, (name, log) in zip(axes, [('1 мин (10 фраз)', log_1min), ('все данные (50 фраз)', log_full)]):
    tr, ev = log[log.split == 'train'], log[log.split == 'eval']
    ax.plot(tr.step, tr.loss_mel_ce.rolling(10, min_periods=1).mean(), label='train (скользящее среднее)')
    ax.plot(ev.step, ev.loss_mel_ce, 'o-', ms=3, label='eval')
    b = ev.loc[ev.loss_mel_ce.idxmin()]
    ax.axvline(b.step, color='gray', ls='--', lw=1)
    ax.set_title(f'{name}: лучший шаг {int(b.step)}'); ax.set_xlabel('шаг'); ax.set_ylabel('loss_mel_ce'); ax.legend()
plt.tight_layout(); plt.show()
