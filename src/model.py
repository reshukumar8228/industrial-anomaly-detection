from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed


def build_lstm_autoencoder(timesteps, n_features):
    inputs = Input(shape=(timesteps, n_features))

    # Encoder
    x = LSTM(32, activation='relu', return_sequences=False)(inputs)
    latent = Dense(16, activation='relu')(x)

    # Decoder
    x = RepeatVector(timesteps)(latent)
    x = LSTM(32, activation='relu', return_sequences=True)(x)
    outputs = TimeDistributed(Dense(n_features))(x)

    model = Model(inputs, outputs)

    model.compile(optimizer='adam', loss='mse')

    return model