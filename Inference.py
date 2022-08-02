import keras
import tensorflow as tf
import numpy as np

print('weights are loaded..')
model = keras.models.load_model('/Users/saima/Desktop/NLP/saarthi_dot_ai/saarthi/checkpoints/weights.h5')

def tokenise_pad_seq(raw_text):
    tokenize = tf.keras.preprocessing.text.Tokenizer()
    tokenize.fit_on_texts(raw_text)
    pre_text = tokenize.texts_to_sequences(raw_text)

    # print('start padding ...')
    max_length = 20
    out_text = tf.keras.preprocessing.sequence.pad_sequences(pre_text, maxlen=max_length)
    return out_text

input_text = 'switch on the lights'
out_text = tokenise_pad_seq(input_text)
y_pred = model.predict(out_text)


labels_action = ['increase', 'decrease', 'activate', 'deactivate', 'change language', 'bring']
labels_object = ['heat', 'lights', 'volume', 'music', 'none', 'lamp', 'newspaper', 'shoes', 'socks', 'juice', 'Chinese', 'Korean', 'German', 'English']
labels_location = ['none', 'washroom', 'kitchen', 'bedroom']


actions = np.array(y_pred[0])
actions = actions.T
objects = np.array(y_pred[1])
objects = objects.T
locations = np.array(y_pred[2])
locations = locations.T

print('action: ', labels_action[np.argmax(actions.argmax(axis=1))])
print('object: ', labels_object[np.argmax(objects.argmax(axis=1))])
print('location:', labels_location[np.argmax(locations.argmax(axis=1))])