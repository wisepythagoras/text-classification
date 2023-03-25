import sys
import numpy as np
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer, tokenizer_from_json
from keras.models import model_from_json
from keras.utils import pad_sequences
from download import clean_str

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

# After we create the model, we need to load the weights for each neuron.
model.load_weights('classifier-weights.h5')
model.make_predict_function()

print(model.summary())


def evaluate(sentence: str, threaded=False) -> list[list[float]]:
    input = tokenizer.texts_to_sequences([sentence])
    input = pad_sequences(input, 140)

    return model.predict(input)


if __name__ == '__main__':
    print('Write some text to classify it. Hit enter on an empty prompt to exit.')

    # Create a prompt to read from the user continually.
    while 1:
        sentence = input('-> ')

        if len(sentence) == 0:
            break

        sentence = clean_str(sentence)
        pred = evaluate(sentence)

        print('')
        print('%s (%f%% confidence)' %
              (labels[np.argmax(pred)], pred[0][np.argmax(pred)] * 100))
        print(pred)
        print('')
