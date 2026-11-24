rows = {'GT (референсы попарно)': {'similarity': gt_c.mean(), 'std': gt_c.std()}}
for k, name in MODELS_C.items():
    ok = np.array([d > 0.02 for d in m_c[k]['sec_per_char']])  # пустые генерации (< 0.02 с на символ)
    rows[name] = {'similarity': sim_c[k].mean(), 'std': sim_c[k].std(),
                  'similarity без пустых': sim_c[k][:, ok].mean(),
                  f'доля пар > {bl["kids"]["mean"]:.3f} (дети)': (sim_c[k] > bl['kids']['mean']).mean(),
                  'CER': m_c[k]['cer_mean'], 'предложений с CER > 0.1': sum(c > 0.1 for c in m_c[k]['cer']),
                  'пустых генераций': int((~ok).sum()), 'F0, Гц': f0_c[k]}
display(pd.DataFrame(rows).T.round(3))
print(f'медиана F0 референсов: {f0_ref:.0f} Гц')
for k in ['1min', 'full']:
    print(MODELS_C[k])
    for i, (c, a) in enumerate(zip(m_c[k]['cer'], m_c[k]['asr'])):
        if c > 0.1:
            print(f'  {i}: CER {c:.2f} | {xe.SENTENCES_RU[i]} -> {a}')
