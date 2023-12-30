import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_and_preprocess_data(vocab_size, maxlen):
    (x_train, y_train), (x_val, y_val) = tf.keras.datasets.imdb.load_data(num_words=vocab_size)
    x_train = pad_sequences(x_train, maxlen=maxlen)
    x_val = pad_sequences(x_val, maxlen=maxlen)
    return x_train, y_train, x_val, y_val
