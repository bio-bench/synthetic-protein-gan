# Synthetic-Protein-GAN

A Generative Adversarial Network (GAN) for generating synthetic protein sequences. This project includes features for customizing the training process by assigning weights to sequences based on keywords in their FASTA headers and filtering sequences by length.

## Features

- **Generative Adversarial Network (GAN)**: Implements a GAN for generating synthetic protein sequences.
- **Keyword-based Sample Weighting**: Assigns higher weights to sequences containing specific keywords in their headers.
- **Length Filtering**: Filters sequences based on minimum and maximum length criteria.
- **Early Stopping**: Includes early stopping to prevent overfitting and optimize training time.

## Installation

To use this project, you will need to have Python installed along with the following libraries:

```bash
pip install numpy tensorflow biopython
