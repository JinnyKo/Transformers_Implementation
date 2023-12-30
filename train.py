import tensorflow as tf
from transformer import create_transformer_model
from data_processing import load_and_preprocess_data

def train_transformer_model():
    vocab_size = 10000
    maxlen = 100
    embed_dim = 32
    num_heads = 2
    ff_dim = 32

    x_train, y_train, x_val, y_val = load_and_preprocess_data(vocab_size, maxlen)

    model = create_transformer_model(maxlen, vocab_size, embed_dim, num_heads, ff_dim)

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    history = model.fit(
        x_train, y_train, batch_size=32, epochs=2, validation_data=(x_val, y_val)
    )

    history_dict = history.history

    predictions = model.predict(x_val)
    return predictions

if __name__ == "__main__":
    train_transformer_model()
