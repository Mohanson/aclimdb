import pickle

import keras.models
import keras.preprocessing
import numpy as np

model = keras.models.load_model('res/aclimdb_model.h5')
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
sequence = keras.preprocessing.sequence.pad_sequences(np.array(sequence), maxlen=400)
y = model.predict(sequence)
print(y[0][0])