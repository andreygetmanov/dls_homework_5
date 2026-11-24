# WavLM-sv как на семинаре: x-vector, L2-нормировка, косинус (xe.SpeakerSim)
# GT: 3 тестовых референса попарно; модели: каждый из 3 референсов против каждой из 10 генераций (30 пар)
# референсы 2 и 3 при генерации не использовались, это проверка на голос, а не на копию референса
sv = xe.SpeakerSim()
gt = sv.refs_pairwise(refs)
sim = {m: sv.matrix(refs, gens[m]) for m in MODELS}
pd.DataFrame({name: sim[m].mean(1) for m, name in MODELS.items()},
             index=[f'референс {i + 1}: {os.path.basename(r)}' for i, r in enumerate(refs)]).round(4)
