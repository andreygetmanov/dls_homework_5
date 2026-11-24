# генерация на Kaggle (kaggle/code/run.py) одной функцией для всех моделей: условие по первому тестовому референсу,
# temperature 0.75, одинаковый сид для каждого предложения, так что модели отличаются только весами GPT
#   model = xe.load_xtts(None | 'runs/1min/best_model.pth' | 'runs/full/best_model.pth')
#   xe.synthesize(model, refs[0], xe.SENTENCES, f'outputs/gens/{name}', name)
refs = [os.path.join(LJS, l.strip()) for l in open(os.path.join(LJS, 'test_refs.txt')) if l.strip()]
MODELS = {'zs': 'zero-shot', '1min': 'few-shot (1 мин)', 'full': 'few-shot (все данные, 5.4 мин)'}
gens = {m: [f'outputs/gens/{m}/{m}_{i:02d}.wav' for i in range(len(xe.SENTENCES))] for m in MODELS}
for i, t in enumerate(xe.SENTENCES):
    print(i, t)
print({m: len([g for g in v if os.path.exists(g)]) for m, v in gens.items()})
