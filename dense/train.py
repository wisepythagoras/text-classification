import json
import numpy as np
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, PReLU
from keras.utils import to_categorical, pad_sequences
from keras import Input, Model


def index_text(text: str) -> list[int]:
    return [dictionary[word] for word in text_to_word_sequence(text)]


# This constant represents the shape of the output. Since we have three different states (negative,
# neutral, positive) we want to instruct our neural network to give us probabilities for just those.
SHAPE = 3

# Extract data from a tsv. We don't need to do anything with the first row.
training = np.genfromtxt(
    '../data/out.tsv', delimiter='\t', skip_header=1,
    usecols=(0, 1), dtype=None)

# The first column contains all the Tweets. Make sure that these are cleaned of special characters.
train_x = [x[0] for x in training]

# The second column contains all the labels.
train_y = np.asarray([x[1] for x in training])

# You can pick whatever number of words you want, but I'll stick with the 4000 most popular words
# from the entire dataset.
max_words = 4000

# The tokenizer will be used to tokenize the tweets and index all words that were found in the entire
# corpus/dataset.
tokenizer = Tokenizer(num_words=max_words)
train_x = [str(x) for x in train_x]

# This line fits the tokenizer on all our text.
tokenizer.fit_on_texts(train_x)

# Once fitted, the tokenizer instance has this `word_index` member which contains the dict to all of
# the words in the dataset.
dictionary = tokenizer.word_index

# You can use this to access it directly and map words.
with open('dictionary.json', 'w') as dictionary_file:
    json.dump(dictionary, dictionary_file)

# Save the tokenizer so that it can be loaded in the predictor.
with open('tokenizer.json', 'w') as tokenizer_file:
    json.dump(json.loads(tokenizer.to_json()), tokenizer_file)

# Save all the word indicies in an array.
all_indices = []

# For each tweet, change each token to its ID in the Tokenizer's word_index.
for text in train_x:
    indices = index_text(text)
    all_indices.append(indices)

# The `max_length` parameter is left empty so that the sequences are padded to the largest.
all_indices = pad_sequences(all_indices)

# Now we have a list of all tweets converted to index arrays.
# Cast as an array for future usage.
all_indices = np.asarray(all_indices)

# The following call converts the sequence of word indecies to arrays of 4000 (one for each word in
# our dictionary) with values of 0 or 1 (if the word was in the tweet or not).
train_x = tokenizer.sequences_to_matrix(all_indices, mode='binary')

# Like previously mentioned, we need to do something similar with the labels. So we create categories.
train_y = to_categorical(train_y, SHAPE)

# You can use a simple Sequential model here as an alternative and instead of the following just run
# `model.add(...)` for each of the following layers. If you use sequential, then you don't need the
# input layer, since the first dense layer has the `input_shape` defined.
inputs = Input(shape=(max_words,))

out = Dense(512, activation='relu', input_shape=(max_words,))(inputs)
out = PReLU()(out)
out = Dropout(0.25)(out)

# I've seen this work without this many hidden layers (eg just with one `Dense(256)`), but I wanted
# to play around. The layers start from 512 units, goes to 256, and finishes off at 128.
for i in range(3):
    out = Dense(512 // (2 ** i), activation='sigmoid')(out)
    out = PReLU()(out)
    out = Dropout(0.5 // (2 ** i))(out)

# This is what I meant in the comment above. Technically you can replace the above array with the
# following layers.
# out = Dense(256, activation='sigmoid')(out)
# out = PReLU()(out)
# out = Dropout(0.5)(out)

# The final layer gives us the output with the desired shape: An array of three predictions.
out = Dense(SHAPE, activation='softmax')(out)

# I like using this method of defining the model, becuase I could expand on my model and add multiple
# outputs to it, but adding more branches to the second parameter.
model = Model(inputs, [out])

# If our output was a simple 0 and 1, then you could use `binary_corssentropy`.
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# It takes a lot of playing with the following values to get your model to function in a specific way.
# You could try tweaking the batch size to 64, increasing or decreasing the epochs, etc, to see what
# results you'll get.
model.fit(train_x, train_y,
          batch_size=32,
          epochs=10,
          verbose=1, # type: ignore
          validation_split=0.1,
          shuffle=True)

# I save the model and the weights here so that I can deploy it and reuse it as I see fit.
model_json = model.to_json()

with open('classifier-model.json', 'w') as json_file:
    json_file.write(model_json)

model.save_weights('classifier-weights.h5')
model.save('classifier-model.h5')

print('Training done')
