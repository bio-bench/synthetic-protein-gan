from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, Flatten

def build_generator(latent_dim, data_shape):
    model = Sequential([
        Dense(128, activation='relu', input_dim=latent_dim),
        Dense(256, activation='relu'),
        Dense(512, activation='relu'),
        Dense(np.prod(data_shape), activation='tanh'),
        Reshape(data_shape)
    ])
    return model

def build_discriminator(data_shape):
    model = Sequential([
        Flatten(input_shape=data_shape),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model
