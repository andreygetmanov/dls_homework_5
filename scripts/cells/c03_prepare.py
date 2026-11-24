# 2.3. подготовьте данные в нужном формате для дообучения (можно использовать скрипты из репозитория)
# формат LJSpeech, как в recipes/ljspeech/xtts_v2: wavs/<id>.wav (22050 Гц, моно) + metadata_*.csv со строками id|text|text
# сами данные готовились скриптами (долго, нужны фильмы):
#   python scripts/smaug_from_srt.py "<фильм>.eng full SDH-Colored.srt"   -> data/segments/smaug
#   python scripts/make_splits.py smaug --min_sim 0.6 --test dos_0827802,dos_1031591,bofa_0061654  -> data/ljs/smaug
# test: 3 фиксированных референса (2 сцены «Пустоши» + «Битва»), в обучение не входят
# eval: 6 фраз для выбора чекпойнта; 1min: случайные фразы из train суммарно >= 60 с
LJS = os.path.join(ROOT, 'data', 'ljs', 'smaug')
split = pd.read_csv(os.path.join(LJS, 'split.csv'))
print(sorted(os.listdir(LJS)))
print(open(os.path.join(LJS, 'metadata_full.csv')).read().splitlines()[:3])
split[['id', 'dur', 'text', 'wer', 'spk_sim', 'split', 'in_1min']].head()
