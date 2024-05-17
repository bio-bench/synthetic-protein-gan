from Bio import SeqIO
import numpy as np

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

def one_hot_encode_sequence(sequence, max_length):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    encoding = np.zeros((max_length, len(amino_acids)))
    for i, aa in enumerate(sequence):
        if aa in amino_acids:
            encoding[i, amino_acids.index(aa)] = 1
    return encoding

def decode_one_hot_sequence(one_hot_sequence):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    decoded_sequence = ''.join([amino_acids[np.argmax(aa)] for aa in one_hot_sequence])
    return decoded_sequence
