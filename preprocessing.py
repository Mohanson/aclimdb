import json
import os
import pickle

import keras.preprocessing.text
import numpy as np

path = '/data/aclImdb'


def get_data(name):
    basename = os.path.basename(name)
    _, score = basename[:-4].split('_')
    score = int(score)
    with open(name, 'r', encoding='utf-8') as f:
        text = f.read().replace('<br />', '')
    return text, score


xtr, ytr, xte, yte = [], [], [], []

print('scan', os.path.join(path, 'train/pos/'))
for entry in os.scandir(os.path.join(path, 'train/pos/')):
    x, y = get_data(entry.path)
    xtr.append(x)
    ytr.append(y)
print('scan', os.path.join(path, 'train/neg/'))
for entry in os.scandir(os.path.join(path, 'train/neg/')):
    x, y = get_data(entry.path)
    xtr.append(x)
    ytr.append(y)
print('scan', os.path.join(path, 'test/pos/'))
for entry in os.scandir(os.path.join(path, 'test/pos/')):
    x, y = get_data(entry.path)
    xte.append(x)
    yte.append(y)
print('scan', os.path.join(path, 'test/neg/'))
for entry in os.scandir(os.path.join(path, 'train/neg/')):
    x, y = get_data(entry.path)
    xte.append(x)
    yte.append(y)

print('dump aclimdb.json')
with open('aclimdb.json', 'w', encoding='utf-8') as f:
    json.dump(((xtr, ytr), (xte, yte)), f)

ytr = [1 if i > 5 else 0 for i in ytr]
yte = [1 if i > 5 else 0 for i in yte]

print('init tokenizer')
tokenizer = keras.preprocessing.text.Tokenizer(num_words=20000)
tokenizer.fit_on_texts(xtr)
print('dump aclimdb_word_index.json')
with open('aclimdb_word_index.json', 'w') as f:
    json.dump(tokenizer.word_index, f)
print('dump aclimdb_index_word.json')
with open('aclimdb_index_word.json', 'w') as f:
    json.dump({v: k for k, v in tokenizer.word_index.items()}, f)
print('dump aclimdb_tokenizer.pkl')
with open('aclimdb_tokenizer.pkl', 'wb') as f:
    pickle.dump(f, tokenizer)

print('serialize xtr')
xtr = tokenizer.texts_to_sequences(xtr)
print('serialize xte')
xte = tokenizer.texts_to_sequences(xte)

print('dump aclimdb.npy')
np.save('aclimdb.npy', np.array([[xtr, ytr], [xte, yte]]))
