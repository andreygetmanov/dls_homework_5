# 2.4. Опишите полученный датасет
print(f'Количество аудиозаписей: {len(split)}')
print(f'Общая длительность: {split.dur.sum() / 60:.2f} мин, средняя длительность записи: {split.dur.mean():.2f} с')

parts = {'train (все данные)': split[split.split == 'full'], 'train (1 мин)': split[split.in_1min],
         'eval': split[split.split == 'eval'], 'test (референсы)': split[split.split == 'test']}
pd.DataFrame({k: {'записей': len(v), 'секунд': round(v.dur.sum(), 1), 'средняя, с': round(v.dur.mean(), 2),
                  'мин, с': v.dur.min(), 'макс, с': v.dur.max()} for k, v in parts.items()}).T
