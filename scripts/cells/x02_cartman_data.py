# данные Картмана: scripts/cartman_lines.py -> --review -> make_splits.py cartman --min_sim 0.75 --test ...
LJS_C = os.path.join(ROOT, 'data', 'ljs', 'cartman')
seg_c = pd.read_csv('data/segments/cartman/segments.csv')
split_c = pd.read_csv(os.path.join(LJS_C, 'split.csv'))
print(f'кандидатов по сценарию: {len(seg_c)} ({seg_c.dur.sum() / 60:.1f} мин); после ручной проверки: '
      f'{(~seg_c["drop"]).sum()} ({seg_c[~seg_c["drop"]].dur.sum() / 60:.1f} мин); '
      f'в датасете (>= 1.5 с, similarity >= 0.75): {len(split_c)} ({split_c.dur.sum() / 60:.1f} мин)')
parts = {'train (все данные)': split_c[split_c.split == 'full'], 'train (1 мин)': split_c[split_c.in_1min],
         'eval': split_c[split_c.split == 'eval'], 'test (референсы)': split_c[split_c.split == 'test']}
display(pd.DataFrame({k: {'записей': len(v), 'секунд': round(v.dur.sum(), 1), 'средняя, с': round(v.dur.mean(), 2)}
                      for k, v in parts.items()}).T)
split_c[['id', 'dur', 'en', 'text', 'spk_sim']].sample(5, random_state=1)
