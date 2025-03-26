# WaveNet Autoencoders for Musical Note Synthesis with Disentangled Representations

This repository contains PyTorch code for training and evaluating WaveNet-based autoencoders for synthesizing musical notes. It includes implementations of both a simplified WaveNet Autoencoder and an enhanced Disentangled WaveNet Autoencoder. The disentangled model aims to learn separate representations for pitch and timbre, allowing for more controllable audio generation.

## Description

This project explores the use of WaveNet architectures for learning representations of musical notes directly from raw audio waveforms.  It provides two main models:

*   **`WaveNetAutoencoder` (Simplified):** A basic WaveNet autoencoder for audio reconstruction. It learns a latent representation of the input audio without explicit conditioning.
*   **`DisentangledWaveNetAE` (Disentangled):** An enhanced WaveNet autoencoder designed to disentangle pitch and timbre. It incorporates pitch conditioning in the decoder and an adversarial component in training to encourage pitch-independent latent representations.

The repository also includes scripts for training these models on a synthetic dataset of musical notes and for evaluating their performance in terms of reconstruction quality, pitch control, timbre preservation, and latent space characteristics.

## Contents

*   **`model.py`:** Contains the PyTorch model definitions for `WaveNetAutoencoder` and `DisentangledWaveNetAE`.
*   **`data.py`:** Defines the `SyntheticNotesDataset` class for generating synthetic musical notes with varying pitches and timbres.
*   **`train.py`:** Includes training loops for both the standard `WaveNetAutoencoder` and the disentangled `DisentangledWaveNetAE`, including adversarial training for the latter.
*   **`eval.py`:** Provides evaluation functions for assessing model performance, including reconstruction quality metrics, pitch control accuracy, timbre similarity measures, and latent space analysis tools.
*   **`demo.ipyn`:** A demo notebook using the modules above.