# дообучение тем же рецептом, --lang ru (kaggle/cartman/code/run.py): 1 мин - 300 шагов, eval каждые 10;
# все данные - 600 шагов, eval каждые 25; чекпойнт с минимальным eval loss_mel_ce на 16 отложенных фразах
def summary_c(name):
    log = pd.read_csv(f'outputs/kaggle_cartman/runs/{name}/log.csv')
    ev = log[log.split == 'eval']
    best = ev.loc[ev.loss_mel_ce.idxmin()]
    print(f'{name}: {int(log.step.max())} шагов за {log.seconds.max() / 60:.1f} мин; eval loss_mel_ce: '
          f'исходная модель {ev.loss_mel_ce.iloc[0]:.3f}, лучший {best.loss_mel_ce:.3f} на шаге {int(best.step)}, '
          f'в конце {ev.loss_mel_ce.iloc[-1]:.3f}')
    return log


fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
for ax, (name, title) in zip(axes, [('1min', '1 мин (17 фраз)'), ('full', 'все данные (144 фразы)')]):
    log = summary_c(name)
    tr, ev = log[log.split == 'train'], log[log.split == 'eval']
    ax.plot(tr.step, tr.loss_mel_ce.rolling(10, min_periods=1).mean(), label='train (скользящее среднее)')
    ax.plot(ev.step, ev.loss_mel_ce, 'o-', ms=3, label='eval')
    b = ev.loc[ev.loss_mel_ce.idxmin()]
    ax.axvline(b.step, color='gray', ls='--', lw=1)
    ax.set_title(f'Картман, {title}: лучший шаг {int(b.step)}'); ax.set_xlabel('шаг'); ax.set_ylabel('loss_mel_ce')
    ax.legend()
plt.tight_layout(); plt.show()
