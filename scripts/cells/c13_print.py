print(f'Speaker similarity GT: {gt.mean():.4f} ± {gt.std():.4f}')
print(f'Speaker similarity zero-shot: {sim["zs"].mean():.4f} ± {sim["zs"].std():.4f}')
print(f'Speaker similarity few-shot (1 мин): {sim["1min"].mean():.4f} ± {sim["1min"].std():.4f}')
print(f'Speaker similarity few-shot (10 мин): {sim["full"].mean():.4f} ± {sim["full"].std():.4f}  (все данные: 5.4 мин, больше речи Смауга нет)')
