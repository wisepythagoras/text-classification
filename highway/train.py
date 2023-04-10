import json
import sys
sys.path.append('.')

import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, LSTM, Conv1D, Embedding, MaxPooling1D, \
                         Bidirectional, Dropout, BatchNormalizationV2
from keras.utils import to_categorical, pad_sequences
from keras.optimizers import Adam
from keras import Input, Model
from math import floor
from highway import Highway

# This constant represents the shape of the output. Since we have three different states (negative,
# neutral, positive) we want to instruct our neural network to give us probabilities for just those.
SHAPE = 3

# Extract data from a tsv. We don't need to do anything with the first row. You can limit the amount
# of data (for testing your model on smaller datasets) by adding `max_rows=X`, where X a positive
# number.
training = np.genfromtxt(
    '../data/out2.tsv', delimiter='\t', skip_header=1,
    usecols=(0, 1), dtype=None)
# np.random.shuffle(training)

# The first column contains all the Tweets. Make sure that these are cleaned of special characters.
train_x = [x[0].lower().strip() for x in training]

# The second column contains all the labels.
train_y = np.asarray([x[1] for x in training])

# In this example I'll test with more words than the dense example.
max_words = 20000
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
inputs = Input(shape=(max_text_len,), name='main_network_input')

# The strategy has changed yet again. There are two convolutional layers that feed into a bidirectional
# LSTM and from there it goes into a Dense layer before the output.
out = Embedding(max_words, 96, input_length=max_text_len, name='input_embeddings')(inputs)
out = Conv1D(96, 3, padding='same', activation='relu', name='pre_highway_conv_filter')(out)
out = MaxPooling1D(pool_size=2, name='post_conv_max_pooling')(out)
out = BatchNormalizationV2(name='batch_normalization_1')(out)
out = Highway(name='main_highway')(out)
# out = Dropout(0.1)(out)
out = BatchNormalizationV2(name='batch_normalization_2')(out)
out = Bidirectional(LSTM(96, dropout=0.2, recurrent_dropout=0.1, activation='relu'), name='main_lstm')(out)
out = BatchNormalizationV2(name='batch_normalization_3')(out)
out = Dense(32, activation='relu', name='dense_pre_output')(out)
# out = Dropout(0.1)(out)

# out = Embedding(max_words, 96, input_length=max_text_len)(inputs)
# out = Conv1D(96, 3, padding='same', activation='relu')(out)
# out = MaxPooling1D(pool_size=2)(out)
# out = BatchNormalizationV2()(out)
# out = Highway()(out)
# out = Bidirectional(LSTM(96, dropout=0.2, recurrent_dropout=0.1, activation='relu'))(out)
# out = Dense(32, activation='relu')(out)
# out = Dropout(0.1)(out)

# The final layer gives us the output with the desired shape: An array of three predictions.
out = Dense(SHAPE, activation='softmax', name='dense_categorical_output')(out)

model = Model(inputs, [out])
print(model.summary())
# import sys; sys.exit()
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=1e-4), # learning_rate=1e-4
              metrics=['accuracy'])

train_amount = floor(len(train_x) * 0.8) # 0.85

# Like in the `dense` example, you can try tweaking the settings here and observe the effects.
model.fit(train_x[:train_amount], train_y[:train_amount],
          batch_size=32,
          epochs=10, # 16
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
