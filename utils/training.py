import numpy as np
import tensorflow as tf

class SequenceDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, sequences, sample_weights, batch_size, latent_dim):
        self.sequences = sequences
        self.sample_weights = sample_weights
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.indexes = np.arange(len(self.sequences))
    
    def __len__(self):
        return len(self.sequences) // self.batch_size
    
    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        return self.__data_generation(batch_indexes)
    
    def __data_generation(self, batch_indexes):
        X = np.random.normal(0, 1, (len(batch_indexes), self.latent_dim))
        y = np.array([self.sequences[i] for i in batch_indexes])
        weights = np.array([self.sample_weights[i] for i in batch_indexes])
        return X, y, weights

def train_gan(generator, discriminator, gan, data_gen, latent_dim, epochs, patience):
    best_g_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        for X, y, weights in data_gen:
            # Train the discriminator
            fake_data = generator.predict(X)
            d_loss_real = discriminator.train_on_batch(y, np.ones((len(y), 1)), sample_weight=weights)
            d_loss_fake = discriminator.train_on_batch(fake_data, np.zeros((len(y), 1)), sample_weight=weights)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # Train the generator
            noise = np.random.normal(0, 1, (len(y), latent_dim))
            valid_y = np.array([1] * len(y))
            g_loss = gan.train_on_batch(noise, valid_y)

        # Early stopping check
        if g_loss < best_g_loss:
            best_g_loss = g_loss
            patience_counter = 0
        else:
            patience_counter += 1

        # Print progress
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Discriminator Loss: {d_loss[0]}, Generator Loss: {g_loss}")

        # Stop training if patience is exceeded
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}.")
            break

def generate_sequences(generator, latent_dim, num_sequences):
    noise = np.random.normal(0, 1, (num_sequences, latent_dim))
    generated_sequences = generator.predict(noise)
    return generated_sequences
