from utils.data_processing import read_fasta_and_weights, one_hot_encode_sequence, decode_one_hot_sequence
from utils.models import build_generator, build_discriminator
from utils.training import train_gan, generate_sequences, SequenceDataGenerator
import numpy as np
import tensorflow as tf

# Ensure TensorFlow uses GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

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
gan_input = tf.keras.layers.Input(shape=(latent_dim,))
fake_data = generator(gan_input)
gan_output = discriminator(fake_data)
gan = tf.keras.Model(gan_input, gan_output)
gan.compile(optimizer='adam', loss='binary_crossentropy')

# Create data generator
data_gen = SequenceDataGenerator(encoded_sequences, sample_weights, batch_size, latent_dim)

# Train GAN with early stopping
train_gan(generator, discriminator, gan, data_gen, latent_dim, epochs, patience)

# Generate and decode new sequences
num_sequences = 10  # Number of new sequences you want to generate
new_sequences = generate_sequences(generator, latent_dim, num_sequences)
decoded_sequences = [decode_one_hot_sequence(seq) for seq in new_sequences]

# Print generated sequences
for seq in decoded_sequences:
    print(seq)
