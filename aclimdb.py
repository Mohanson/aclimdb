import pickle
import random

import keras.callbacks
import keras.layers
import keras.models
import keras.preprocessing.sequence
import keras.preprocessing.text
import numpy as np


def create_ngram_set(input_list, ngram_value=2):
    """
    Extract a set of n-grams from a list of integers.

    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
    {(4, 9), (4, 1), (1, 4), (9, 4)}

    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=3)
    [(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)]
    """
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))


def add_ngram(sequences, token_indice, ngram_range=2):
    """
    Augment the input list of list (sequences) by appending n-grams values.

    Example: adding bi-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
    >>> add_ngram(sequences, token_indice, ngram_range=2)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]

    Example: adding tri-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017, (7, 9, 2): 2018}
    >>> add_ngram(sequences, token_indice, ngram_range=3)
    [[1, 3, 4, 5, 1337], [1, 3, 7, 9, 2, 1337, 2018]]
    """
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for i in range(len(new_list) - ngram_range + 1):
            for ngram_value in range(2, ngram_range + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)

    return new_sequences


# Set parameters:
# ngram_range = 2 will add bi-grams features
ngram_range = 1
max_features = 20000
maxlen = 400
batch_size = 32
embedding_dims = 50
epochs = 20

print('Loading data...')
with open('aclimdb.npy', 'rb') as f:
    (x_train, y_train), (x_test, y_test) = np.load(f)

print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
print('Average train sequence length: {}'.format(np.mean(list(map(len, x_train)), dtype=int)))
print('Average test sequence length: {}'.format(np.mean(list(map(len, x_test)), dtype=int)))

if ngram_range > 1:
    print('Adding {}-gram features'.format(ngram_range))
    # Create set of unique n-gram from the training set.
    ngram_set = set()
    for input_list in x_train:
        for i in range(2, ngram_range + 1):
            set_of_ngram = create_ngram_set(input_list, ngram_value=i)
            ngram_set.update(set_of_ngram)

    # Dictionary mapping n-gram token to a unique integer.
    # Integer values are greater than max_features in order
    # to avoid collision with existing features.
    start_index = max_features + 1
    token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
    indice_token = {token_indice[k]: k for k in token_indice}

    # max_features is the highest integer that could be found in the dataset.
    max_features = np.max(list(indice_token.keys())) + 1

    # Augmenting x_train and x_test with n-grams features
    x_train = add_ngram(x_train, token_indice, ngram_range)
    x_test = add_ngram(x_test, token_indice, ngram_range)
    print('Average train sequence length: {}'.format(np.mean(list(map(len, x_train)), dtype=int)))
    print('Average test sequence length: {}'.format(np.mean(list(map(len, x_test)), dtype=int)))

print('Pad sequences (samples x time)')
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
model = keras.models.Sequential()
model.add(keras.layers.Embedding(max_features, embedding_dims, input_length=maxlen))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


def maintr():
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)],
              validation_data=(x_test, y_test))
    model.save_weights('aclimdb_weights.h5')


def mainte():
    model.load_weights('aclimdb_weights.h5')
    with open('aclimdb_tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    text = """\
    This must be one of the most horribly titled films of all time. The kind of title that ruins a film because it
    neither evokes the plot nor the characters. A title like this makes a film flop, even the French title is not
    much better. Too bad - Truffaut & Deneuve must have been enough to sell it..This is a long film, but largely worth
    it. Clearly influenced by Hitchcock, we have an intercontinental story about a personal ad bride, her rich
    husband, a theft, an identity switch, and obsessive love. The plot here is actually very good, and takes us on
    an unexpected trip.The thing that works both for and against the movie is the focus on the relationship. It is
    an interesting study in how these plot developments are played out in "real life relationship" with these two
    people. Unfortunately, this is what bogs the film down, and makes it ultimately dissatisfying. We do like films
    to have a real sense of finality, and that is missing here.It was the case in many of her films that Deneuve
    became a canvas for Directors to play their fantasies out on, and this time it doesn\'t work as well. Messy here,
    is the fact that the Director clearly just wanted to have Deneuve take her top off a few times. Deneuve is an
    actress who always seems very deliberate and thoughtful, so these attempts to make her seem spontaneous fall flat.
    Basically, the script needed to be worked out better before shooting began, to make this film tighter and shorter
    and to snap. But Truffaut didn\'t snap, did he? So - it wanders a bit, but remains interesting."""
    sequence = tokenizer.texts_to_sequences([text])
    sequence = keras.preprocessing.sequence.pad_sequences(np.array(sequence), maxlen=maxlen)
    y = model.predict(sequence)
    print(y[0][0])


if __name__ == '__main__':
    maintr( )
