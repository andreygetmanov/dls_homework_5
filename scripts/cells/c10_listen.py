################ Выведите в этой ячейке gt запись и по одой записи от каждой модели
ref_text = split.set_index('id').loc[os.path.basename(refs[0])[:-4], 'text']
print('GT (референс для генерации):', ref_text)
display(Audio(refs[0]))
k = 8
print(f'«{xe.SENTENCES[k]}»')
for m, name in MODELS.items():
    print(name)
    display(Audio(gens[m][k]))
