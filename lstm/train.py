import json
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, LSTM, Conv1D, Embedding, MaxPooling1D
from keras.utils import to_categorical, pad_sequences
from keras import Input, Model
from math import floor

# This constant represents the shape of the output. Since we have three different states (negative,
# neutral, positive) we want to instruct our neural network to give us probabilities for just those.
SHAPE = 3

# Extract data from a tsv. We don't need to do anything with the first row. You can limit the amount
# of data (for testing your model on smaller datasets) by adding `max_rows=X`, where X a positive
# number.
training = np.genfromtxt(
    '../data/out.tsv', delimiter='\t', skip_header=1,
    usecols=(0, 1), dtype=None)

# The first column contains all the Tweets. Make sure that these are cleaned of special characters.
train_x = [x[0] for x in training]

# The second column contains all the labels.
train_y = np.asarray([x[1] for x in training])

# In this example I'll test with more words than the dense example.
max_words = 8000
max_text_len = 140

# The tokenizer will be used to tokenize the tweets and index all words that were found in the entire
# corpus/dataset.
tokenizer = Tokenizer(num_words=max_words)
train_x = [str(x) for x in train_x]

# This line fits the tokenizer on all our text.
tokenizer.fit_on_texts(train_x)

# Once fitted, the tokenizer instance has this `word_index` member which contains the dict to all of
# the words in the dataset.
dictionary = tokenizer.word_index

# Save the tokenizer so that it can be loaded in the predictor. I don't save the dictionary, as I do
# not use it at all.
with open('tokenizer.json', 'w') as tokenizer_file:
    json.dump(json.loads(tokenizer.to_json()), tokenizer_file)

# Here I create sequences from the text, based on the dictionary. Then I pad all sequences to the max
# allowed length of text.
train_x = tokenizer.texts_to_sequences(train_x)
train_x = pad_sequences(train_x, maxlen=max_text_len)

# Like previously mentioned, we need to do something similar with the labels. So we create categories.
train_y = to_categorical(train_y, SHAPE)

# I follow the exact same pattern as the `dense` example.
inputs = Input(shape=(max_text_len,))

# In this example I use a convolutional 1D layer to pass it down to an LSTM with 100 memory units. I
# played with the dropouts as well, but it seemed to make the results worse, so I removed them.
out = Embedding(max_words, 64, input_length=max_text_len)(inputs)
out = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(out)
out = MaxPooling1D(pool_size=2)(out)
out = LSTM(100)(out)

# The final layer gives us the output with the desired shape: An array of three predictions.
out = Dense(SHAPE, activation='softmax')(out)

model = Model(inputs, [out])
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

train_amount = floor(len(train_x) * 0.7)

# Like in the `dense` example, you can try tweaking the settings here and observe the effects.
model.fit(train_x[:train_amount], train_y[:train_amount],
          batch_size=64,
          epochs=10,
          verbose=1, # type: ignore
          validation_data=(train_x[-train_amount + 1:], train_y[-train_amount + 1:]),
          shuffle=True)

# I save the model and the weights here so that I can deploy it and reuse it as I see fit.
model_json = model.to_json()

with open('classifier-model.json', 'w') as json_file:
    json_file.write(model_json)

model.save_weights('classifier-weights.h5')
model.save('classifier-model.h5')

print('Training done')
