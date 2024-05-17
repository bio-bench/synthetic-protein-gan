import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten
from tensorflow.keras.models import Model, Sequential
from Bio import SeqIO

# Verify that TensorFlow is using the GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Function to build the generator model
def build_generator(latent_dim, data_shape):
    model = Sequential([
        Dense(128, activation='relu', input_dim=latent_dim),
        Dense(256, activation='relu'),
        Dense(512, activation='relu'),
        Dense(np.prod(data_shape), activation='tanh'),
        Reshape(data_shape)
    ])
    return model

# Function to build the discriminator model
def build_discriminator(data_shape):
    model = Sequential([
        Flatten(input_shape=data_shape),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model

# Function to read sequences and calculate sample weights based on headers with multiple keywords
def read_fasta_and_weights(file_path, keywords=["Homo sapiens", "mus", "ratus"], weight_factor=10, min_length=50, max_length=1000, encoding='utf-8'):
    sequences = []
    sample_weights = []
    with open(file_path, 'r', encoding=encoding) as handle:
        for record in SeqIO.parse(handle, "fasta"):
            seq_length = len(record.seq)
            if min_length <= seq_length <= max_length:
                sequences.append(str(record.seq))
                if any(keyword in record.description for keyword in keywords):
                    sample_weights.append(weight_factor)
                else:
                    sample_weights.append(1)
    return sequences, np.array(sample_weights)

# Function to one-hot encode a single sequence
def one_hot_encode_sequence(sequence, max_length):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    encoding = np.zeros((max_length, len(amino_acids)))
    for i, aa in enumerate(sequence):
        if aa in amino_acids:
            encoding[i, amino_acids.index(aa)] = 1
    return encoding

# Function to decode one-hot encoded sequence back to amino acid sequence
def decode_one_hot_sequence(one_hot_sequence):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    decoded_sequence = ''.join([amino_acids[np.argmax(aa)] for aa in one_hot_sequence])
    return decoded_sequence

# Function to generate new sequences
def generate_sequences(generator, latent_dim, num_sequences):
    noise = np.random.normal(0, 1, (num_sequences, latent_dim))
    generated_sequences = generator.predict(noise)
    return generated_sequences

# Define file path and parameters
file_path = 'data/your_sequences.fasta'  # Path to your plain FASTA file
keywords = ["OS=Homo sapiens", "OS=mus", "OS=ratus"]
weight_factor = 10
min_length = 50  # Minimum length of sequences to keep
max_length = 1000  # Maximum length of sequences to keep
latent_dim = 100
batch_size = 32
epochs = 10000
patience = 10  # Number of epochs to wait for improvement before stopping

# Read and preprocess sequences
sequences, sample_weights = read_fasta_and_weights(file_path, keywords, weight_factor, min_length, max_length)
max_sequence_length = max(len(seq) for seq in sequences)
encoded_sequences = [one_hot_encode_sequence(seq, max_sequence_length) for seq in sequences]
encoded_sequences = np.array(encoded_sequences)

# Build and compile models
generator = build_generator(latent_dim, (max_sequence_length, 20))
discriminator = build_discriminator((max_sequence_length, 20))
discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define and compile GAN
discriminator.trainable = False
gan_input = Input(shape=(latent_dim,))
fake_data = generator(gan_input)
gan_output = discriminator(fake_data)
gan = Model(gan_input, gan_output)
gan.compile(optimizer='adam', loss='binary_crossentropy')

# Custom Data Generator
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

data_gen = SequenceDataGenerator(encoded_sequences, sample_weights, batch_size, latent_dim)

# Function to train GAN with early stopping
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

# Start training with early stopping
train_gan(generator, discriminator, gan, data_gen, latent_dim, epochs, patience)

# Generate and decode new sequences
num_sequences = 10  # Number of new sequences you want to generate
new_sequences = generate_sequences(generator, latent_dim, num_sequences)
decoded_sequences = [decode_one_hot_sequence(seq) for seq in new_sequences]

# Print generated sequences
for seq in decoded_sequences:
    print(seq)
