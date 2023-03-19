import sys
import numpy as np
from keras.preprocessing.text import Tokenizer, text_to_word_sequence, tokenizer_from_json
from keras.models import model_from_json
import tensorflow as tf

labels = ['negative', 'neutral', 'positive']
tokenizer: Tokenizer | None = None

with open('tokenizer.json', 'r') as t:
    tokenizer = tokenizer_from_json(t.read())

json_file = open('classifier-model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

# Load the model and the weights.
model = model_from_json(loaded_model_json)

if model is None:
    print('No trained model')
    sys.exit(1)

model.load_weights('classifier-weights.h5')
model.make_predict_function()

print(model.summary())


def evaluate(sentence: str) -> list[list[float]]:
    words = text_to_word_sequence(sentence)
    inp = tokenizer.texts_to_sequences([words])
    inp = tokenizer.sequences_to_matrix(inp, mode='binary')

    return model.predict(inp)


if __name__ == '__main__':
    print('Input a sentence to be evaluated, or Enter to quit.')

    # Create a prompt to read from the user continually.
    while 1:
        sentence = input('-> ')

        if len(sentence) == 0:
            break

        pred = evaluate(sentence)

        print('')
        print('%s (%f%% confidence)' %
              (labels[np.argmax(pred)], pred[0][np.argmax(pred)] * 100))
        print(pred)
        print('')
