from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense


def build_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))

    # Encoder
    x = Dense(16, activation='relu')(input_layer)
    x = Dense(8, activation='relu')(x)
    latent = Dense(4, activation='relu')(x)

    # Decoder
    x = Dense(8, activation='relu')(latent)
    x = Dense(16, activation='relu')(x)
    output_layer = Dense(input_dim, activation='linear')(x)

    autoencoder = Model(inputs=input_layer, outputs=output_layer)

    autoencoder.compile(
        optimizer='adam',
        loss='mse'
    )

    return autoencoder