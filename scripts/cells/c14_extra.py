m = json.load(open('outputs/metrics.json'))
rows = {}
for k, name in MODELS.items():
    rows[name] = {'similarity': sim[k].mean(), 'доля пар > 0.86': (sim[k] > 0.86).mean(), 'CER': m[k]['cer_mean'],
                  'предложений с CER > 0.1': sum(c > 0.1 for c in m[k]['cer']), 'с/символ': np.mean(m[k]['sec_per_char'])}
display(pd.DataFrame(rows).T.round(3))
for k in ['1min', 'full']:
    print(MODELS[k])
    for i, (c, a) in enumerate(zip(m[k]['cer'], m[k]['asr'])):
        if c > 0.1:
            print(f'  {i}: CER {c:.2f} | {xe.SENTENCES[i]} -> {a}')
