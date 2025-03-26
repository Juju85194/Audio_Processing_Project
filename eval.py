import torch
import numpy as np
from IPython.display import Audio
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate_model(model, dataset, num_samples=3):
    model.eval()
    with torch.no_grad():
        for i in range(num_samples):
            # Get a sample
            audio, midi = dataset[i]
            original_audio = audio.numpy()
            
            # Prepare input
            audio = audio.unsqueeze(0).unsqueeze(0).to(device)  # Add batch and channel dims
            
            # Reconstruct
            reconstructed = model(audio)
            reconstructed_audio = reconstructed.squeeze().cpu().numpy()
            
            # Plot
            plt.figure(figsize=(12, 6))
            plt.subplot(2, 1, 1)
            plt.plot(original_audio)
            plt.title(f'Original Audio (MIDI: {midi.item() + dataset.min_midi})')
            
            plt.subplot(2, 1, 2)
            plt.plot(reconstructed_audio)
            plt.title('Reconstructed Audio')
            
            plt.tight_layout()
            plt.show()
            
            # Play audio
            print("Original:")
            display(Audio(original_audio, rate=16000))
            print("Reconstructed:")
            display(Audio(reconstructed_audio, rate=16000))

def evaluate_with_pitch_control(model, dataset, num_samples=3):
    model.eval()
    with torch.no_grad():
        for i in range(num_samples):
            # Get a sample
            audio, original_midi = dataset[i]
            original_audio = audio.numpy()
            
            # Prepare input
            audio = audio.unsqueeze(0).unsqueeze(0).to(device)
            original_midi = original_midi.to(device)
            
            # Get latent code
            z = model.encode(audio)
            
            # Reconstruct with original pitch
            reconstructed = model.decode(z, original_midi)
            reconstructed_audio = reconstructed.squeeze().cpu().numpy()
            
            # Generate with different pitch
            new_pitch = torch.randint(0, 88, (1,)).to(device)
            modified = model.decode(z, new_pitch)
            modified_audio = modified.squeeze().cpu().numpy()
            
            # Plot
            plt.figure(figsize=(12, 8))
            
            plt.subplot(3, 1, 1)
            plt.plot(original_audio)
            plt.title(f'Original Audio (MIDI: {original_midi.item() + dataset.min_midi})')
            
            plt.subplot(3, 1, 2)
            plt.plot(reconstructed_audio)
            plt.title('Reconstructed (Same Pitch)')
            
            plt.subplot(3, 1, 3)
            plt.plot(modified_audio)
            plt.title(f'Modified Pitch (MIDI: {new_pitch.item() + dataset.min_midi})')
            
            plt.tight_layout()
            plt.show()
            
            # Play audio
            print("Original:")
            display(Audio(original_audio, rate=16000))
            print("Reconstructed (same pitch):")
            display(Audio(reconstructed_audio, rate=16000))
            print(f"Modified (new pitch {new_pitch.item() + dataset.min_midi}):")
            display(Audio(modified_audio, rate=16000))