# генерация на Kaggle (kaggle/cartman/code/run.py): xe.synthesize(model, refs_c[0], xe.SENTENCES_RU, ..., lang='ru')
# метрики: python scripts/evaluate.py cartman zs 1min full -> outputs/metrics_cartman.json (CER по whisper)
import librosa

refs_c = [os.path.join(LJS_C, l.strip()) for l in open(os.path.join(LJS_C, 'test_refs.txt')) if l.strip()]
MODELS_C = {'zs': 'zero-shot', '1min': 'few-shot (1 мин)', 'full': 'few-shot (все данные, 8.6 мин)'}
gens_c = {m: [f'outputs/gens_cartman/{m}/{m}_{i:02d}.wav' for i in range(len(xe.SENTENCES_RU))] for m in MODELS_C}
m_c = json.load(open('outputs/metrics_cartman.json'))
gt_c = sv.refs_pairwise(refs_c)
sim_c = {m: sv.matrix(refs_c, gens_c[m]) for m in MODELS_C}  # пустая генерация дополняется тишиной до 0.5 с


def f0(path):
    y, _ = librosa.load(path, sr=16000)
    if len(y) < 8000:
        return np.nan
    f, v, _ = librosa.pyin(y, fmin=80, fmax=700, sr=16000)
    return np.nanmedian(f[v]) if v.any() else np.nan


f0_ref = np.median([f0(r) for r in refs_c])
f0_c = {m: np.nanmedian([f0(g) for g in gens_c[m]]) for m in MODELS_C}
print(f'similarity: GT {gt_c.mean():.4f}; zero-shot {sim_c["zs"].mean():.4f} ± {sim_c["zs"].std():.4f} '
      f'(другие дети озвучки {bl["kids"]["mean"]:.3f}, другие фразы Картмана {bl["cartman"]["mean"]:.3f})')
print(f'CER zero-shot {m_c["zs"]["cer_mean"]:.3f}; медиана F0: референсы {f0_ref:.0f} Гц, zero-shot {f0_c["zs"]:.0f} Гц')
display(Audio(refs_c[0]))
k_c = 6
print(f'«{xe.SENTENCES_RU[k_c]}», zero-shot:')
display(Audio(gens_c['zs'][k_c]))
