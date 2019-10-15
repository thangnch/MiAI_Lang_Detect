import pandas as pd
# Import Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import os.path
from os import path
import re
from distutils.version import LooseVersion
import warnings
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy
import pickle
import sys

# Hyperparameters
max_sentence_length = 200
embedding_vector_length = 300
dropout = 0.5

def process_sentence(sentence):
    # Loai bo cac ki tu dac biet, chuyen cau ve lower case
    return re.sub(r'[\\\\/:*«`\'?¿";!<>,.|]', '', sentence.lower().strip())


def create_lookup_tables(text):
    # Tao bang tra cuu Vocab
    #:param text: Text duoc chia nho thanh cac word
    #:return:  (vocab_to_int, int_to_vocab)

    vocab = set(text)

    vocab_to_int = {word: i for i, word in enumerate(vocab)}
    int_to_vocab = {v: k for k, v in vocab_to_int.items()}

    return vocab_to_int, int_to_vocab


def convert_to_int(data, data_int):
    # Chuyen doi text thanh vector so
    all_items = []
    for sentence in data:
        all_items.append([data_int[word] if word in data_int else data_int["<UNK>"] for word in sentence.split()])

    return all_items

def load_data():
    # Ham load du lieu tu file data.csv
    data = pd.read_csv("data.csv", names=["sentence", "language"], header=None)
    print(data.describe())
    return data

def get_model():
    # Ham tao model
    model = Sequential()

    model.add(Embedding(len(vocab_to_int), embedding_vector_length, input_length=max_sentence_length))
    model.add(LSTM(256, return_sequences=True, dropout=dropout, recurrent_dropout=dropout))
    model.add(LSTM(256, dropout=dropout, recurrent_dropout=dropout))
    model.add(Dense(len(languages), activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def predict_sentence(model, sentence,  vocab_to_int, int_to_languages):
    # Chuyen text thanh vector int va dua vao model de predict language

    # Clean the sentence
    sentence = process_sentence(sentence)

    # Transform and pad it before using the model to predict
    x = numpy.array(convert_to_int([sentence], vocab_to_int))
    x = sequence.pad_sequences(x, maxlen=max_sentence_length)

    prediction = model.predict(x)
    #print(prediction[0])

    # Get the highest prediction
    lang_index = numpy.argmax(prediction)
    print(prediction[0][lang_index])

    # Neu probality <0.3 thi hien thi ngon ngu Khong xac dinh/Unknown
    if prediction[0][lang_index]<0.3:
        return "Unknown"
    else:
        return int_to_languages[lang_index]

mode = "test"
if len(sys.argv)>1:
    mode = sys.argv[1]

if mode=="test":
    predSentence = sys.argv[2]

# Neu mode la train thi thuc hien train
if mode=="train":

    data = load_data()

    # Xao tron data
    sss = StratifiedShuffleSplit(test_size=0.2, random_state=0)

    # Lam sach cau
    X = data["sentence"].apply(process_sentence)
    y = data["language"]

    # Chia data thanh cac cau
    elements = (' '.join([sentence for sentence in X])).split()

    X_train, X_test, y_train, y_test = None, None, None, None

    # Chia du lieu train, test
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    # Lay danh sach cac ngon ngu
    languages = set(y)

    # Them word UNK cho cac word ko co trong vocab
    elements.append("<UNK>")

    # Chuyen vocab thanh vector so
    vocab_to_int, int_to_vocab = create_lookup_tables(elements)
    languages_to_int, int_to_languages = create_lookup_tables(y)

    # Luu lai cac du lieu
    with open("data.pkl", "wb") as fp:
        pickle.dump([vocab_to_int, int_to_vocab, languages_to_int, int_to_languages, languages] , fp)

    # Encode du lieu X
    X_test_encoded = convert_to_int(X_test, vocab_to_int)
    X_train_encoded = convert_to_int(X_train, vocab_to_int)

    # Encode du lieu Y
    y_data = convert_to_int(y_test, languages_to_int)

    # Tao encode one hot
    enc = OneHotEncoder()
    enc.fit(y_data)

    # Chuyen du lieu y thanh Encode
    y_train_encoded = enc.fit_transform(convert_to_int(y_train, languages_to_int)).toarray()
    y_test_encoded = enc.fit_transform(convert_to_int(y_test, languages_to_int)).toarray()

    # Check TensorFlow Version
    assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer'
    print('TensorFlow Version: {}'.format(tf.__version__))

    # Check for a GPU
    if not tf.test.gpu_device_name():
        warnings.warn('No GPU found. Please use a GPU to train your neural network.')
    else:
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))




    #with tf.device('/gpu:0'):
        # Pad cac dau cho du do dai
    X_train_pad = sequence.pad_sequences(X_train_encoded, maxlen=max_sentence_length)
    X_test_pad = sequence.pad_sequences(X_test_encoded, maxlen=max_sentence_length)

    # Tao model
    model = get_model()
    print(model.summary())

    # Train the model
    model.fit(X_train_pad, y_train_encoded, epochs=5, batch_size=256)

    # Danh gia model
    scores = model.evaluate(X_test_pad, y_test_encoded, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

    # Luu Model vao file
    model.save("model.h5")
    print("Model trained and saved!")
else:
    # Doc du lieu tu file
    with open("data.pkl", "rb") as fp:  # Unpickling
        vocab_to_int, int_to_vocab, languages_to_int, int_to_languages, languages = pickle.load(fp)

    print("Vocab loaded!")

    # Doc model tu file
    model = get_model()
    model.load_weights("model.h5")
    print("Model loaded!")


    # Predict
    print("Check language for sentence= ", predSentence)
    print("Language=", predict_sentence(model, predSentence, vocab_to_int, int_to_languages))