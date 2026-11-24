print('GT (референс для генерации):', split_c.set_index('id').loc[os.path.basename(refs_c[0])[:-4], 'text'])
display(Audio(refs_c[0]))
k_c = 8  # предложение, которое все три модели сказали без ошибок: сравнение только голоса
print(f'«{xe.SENTENCES_RU[k_c]}»')
for m, name in MODELS_C.items():
    print(name)
    display(Audio(gens_c[m][k_c]))
