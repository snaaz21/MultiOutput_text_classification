import pandas as pd
from keras.models import Model
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from sklearn.model_selection import train_test_split
from keras.layers import Input
import keras
import tensorflow as tf
import re
from nltk.corpus import stopwords

def main(train_csv, epochs):
    df = pd.read_csv(train_csv)

    # print(df.columns)
    # print(df['action'].value_counts())
    # print(df['object'].value_counts())
    # print(df['location'].value_counts())

    df = df.reset_index(drop=True)
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
    STOPWORDS = set(stopwords.words('english'))

    def clean_text(text):
        """
            text: a string

            return: modified initial string
        """
        text = text.lower()  # lowercase text
        text = REPLACE_BY_SPACE_RE.sub(' ', text)
        text = BAD_SYMBOLS_RE.sub('', text)
        text = text.replace('x', '')
        # text = re.sub(r'\W+', '', text)
        text = ' '.join(word for word in text.split() if word not in STOPWORDS)  # remove stopwors from text
        return text

    df['transcription'] = df['transcription'].apply(clean_text)
    df['action'] = df['action'].apply(clean_text)
    df['object'] = df['object'].apply(clean_text)
    df['location'] = df['location'].apply(clean_text)
    print(df['transcription'])

    MAX_NB_WORDS = 5000
    MAX_SEQUENCE_LENGTH = 250
    EMBEDDING_DIM = 100

    x = df['transcription']

    tokenize = tf.keras.preprocessing.text.Tokenizer()
    tokenize.fit_on_texts(x)
    x = tokenize.texts_to_sequences(x)

    print('start padding ...')
    max_length = 20
    x = tf.keras.preprocessing.sequence.pad_sequences(x, maxlen=max_length)
    print(x)

    Y = df[["action", "object", "location"]]
    X_train, X_test, Y_train, Y_test = train_test_split(x, Y, test_size=0.20, random_state=42)

    print(X_train.shape, Y_train.shape)
    print(X_test.shape, Y_test.shape)
    y1_train = pd.get_dummies(Y_train[["action"]]).values
    y1_test = pd.get_dummies(Y_test[["action"]]).values

    y2_train = pd.get_dummies(Y_train[["object"]]).values
    y2_test = pd.get_dummies(Y_test[["object"]]).values

    y3_train = pd.get_dummies(Y_train[["location"]]).values
    y3_test = pd.get_dummies(Y_test[["location"]]).values

    maxlen = 20  # MAX_SEQUENCE_LENGTH
    input_1 = Input(shape=(maxlen,))
    embedding_layer = Embedding(MAX_NB_WORDS, EMBEDDING_DIM)(
        input_1)  # Embedding(vocab_size, 100, weights=[embedding_matrix], trainable=False)(input_1)
    LSTM_Layer1 = LSTM(128)(embedding_layer)

    output1 = Dense(6, activation='softmax')(LSTM_Layer1)
    output2 = Dense(14, activation='softmax')(LSTM_Layer1)
    output3 = Dense(4, activation='softmax')(LSTM_Layer1)

    model = Model(inputs=input_1, outputs=[output1, output2, output3])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    print(model.summary())
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")

    model.fit(x=X_train, y=[y1_train, y2_train, y3_train], batch_size=1, epochs=epochs, verbose=1,
                        validation_split=0.2, callbacks=[tensorboard_callback,
                                                         keras.callbacks.EarlyStopping(monitor='val_loss', patience=3,
                                                                                       min_delta=0.0001)])

    model.save('weights.h5')
    yaml_model = model.to_yaml()
    with open('model_config.yaml', 'w') as yaml_file:
        yaml_file.write(yaml_model)
    yaml_file.close()

    score = model.evaluate(x=X_test, y=[y1_test, y2_test, y3_test], verbose=1)
    print("Test Score:", score[0])
    print("Test Accuracy:", score[1])

    print('after loading...')
    model = keras.models.load_model('weights.h5')
    score = model.evaluate(x=X_test, y=[y1_test, y2_test, y3_test], verbose=1)

    print("Test Score:", score[0])
    print("Test Accuracy:", score[1])

if __name__ == '__main__':
    train_csv_path = '/Users/saima/Desktop/orgs/saarithi_dot_ai/task_data/train_data.csv'
    epochs = 2
    main(train_csv_path, epochs)



