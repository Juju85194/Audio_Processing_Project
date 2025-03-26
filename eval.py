import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from IPython.display import Audio
import matplotlib.pyplot as plt
from tqdm import tqdm

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
            
            # Generate with different pitch (more extreme change)
            pitch_change = 24  # Change by 2 octaves to make it more noticeable
            new_pitch = torch.clamp(original_midi + pitch_change, 0, 87)  # Keep within MIDI range
            modified = model.decode(z, new_pitch)
            modified_audio = modified.squeeze().cpu().numpy()
            
            # Calculate actual frequency ratios
            def estimate_fundamental_freq(audio_np):
                fft = np.fft.rfft(audio_np)
                freqs = np.fft.rfftfreq(len(audio_np), 1/16000)
                magnitudes = np.abs(fft)
                # Skip DC component and find peak
                peak_idx = np.argmax(magnitudes[1:]) + 1
                return freqs[peak_idx]
            
            orig_freq = estimate_fundamental_freq(original_audio)
            recon_freq = estimate_fundamental_freq(reconstructed_audio)
            modified_freq = estimate_fundamental_freq(modified_audio)
            
            expected_ratio = 2 ** (pitch_change / 12)  # Expected frequency ratio
            
            # Plot
            plt.figure(figsize=(12, 10))
            
            plt.subplot(3, 1, 1)
            plt.plot(original_audio)
            plt.title(f'Original Audio (MIDI: {original_midi.item() + 21}, Freq: {orig_freq:.1f}Hz)')
            
            plt.subplot(3, 1, 2)
            plt.plot(reconstructed_audio)
            plt.title(f'Reconstructed (MIDI: {original_midi.item() + 21}, Freq: {recon_freq:.1f}Hz)')
            
            plt.subplot(3, 1, 3)
            plt.plot(modified_audio)
            plt.title(f'Modified (MIDI: {new_pitch.item() + 21}, Freq: {modified_freq:.1f}Hz)\n'
                     f'Expected freq ratio: {expected_ratio:.2f}, Actual ratio: {modified_freq/recon_freq:.2f}')
            
            plt.tight_layout()
            plt.show()
            
            # Play audio
            print("Original:")
            display(Audio(original_audio, rate=16000))
            print("Reconstructed (same pitch):")
            display(Audio(reconstructed_audio, rate=16000))
            print(f"Modified (new pitch {new_pitch.item() + 21}):")
            display(Audio(modified_audio, rate=16000))

def compare_models(original_model, disentangled_model, dataset, num_samples):
    """
    Compare the original and disentangled models across multiple metrics
    """
    results = {}
    
    # Set models to evaluation mode
    original_model.eval()
    disentangled_model.eval()
    
    # 1. Basic Reconstruction Quality
    print("\n=== Reconstruction Quality ===")
    recon_losses = {'original': [], 'disentangled': []}
    with torch.no_grad():
        for i in tqdm(range(num_samples), desc="Evaluating reconstruction"):
            audio, midi = dataset[i]
            audio = audio.unsqueeze(0).unsqueeze(0).to(device)
            midi = midi.to(device)
            
            # Original model
            orig_recon = original_model(audio)
            orig_loss = F.mse_loss(orig_recon, audio)
            recon_losses['original'].append(orig_loss.item())
            
            # Disentangled model
            dis_recon = disentangled_model.decode(disentangled_model.encode(audio), midi)
            dis_loss = F.mse_loss(dis_recon, audio)
            recon_losses['disentangled'].append(dis_loss.item())
            
            # Plot first sample
            if i == 0:
                plt.figure(figsize=(12, 6))
                plt.plot(audio.squeeze().cpu().numpy(), label='Original', alpha=0.7)
                plt.plot(orig_recon.squeeze().cpu().numpy(), label='Original Model', alpha=0.7)
                plt.plot(dis_recon.squeeze().cpu().numpy(), label='Disentangled Model', alpha=0.7)
                plt.legend()
                plt.title("Reconstruction Comparison")
                plt.show()
                
                print("Original Audio:")
                display(Audio(audio.squeeze().cpu().numpy(), rate=16000))
                print("Original Model Reconstruction:")
                display(Audio(orig_recon.squeeze().cpu().numpy(), rate=16000))
                print("Disentangled Model Reconstruction:")
                display(Audio(dis_recon.squeeze().cpu().numpy(), rate=16000))
    
    results['reconstruction_loss'] = {
        'original': np.mean(recon_losses['original']),
        'disentangled': np.mean(recon_losses['disentangled'])
    }
    print(f"\nAverage Reconstruction Loss:")
    print(f"Original: {results['reconstruction_loss']['original']:.6f}")
    print(f"Disentangled: {results['reconstruction_loss']['disentangled']:.6f}")
    
    # 2. Pitch Control Accuracy
    print("\n=== Pitch Control Accuracy ===")
    pitch_accuracies = {'original': [], 'disentangled': []}
    with torch.no_grad():
        for i in tqdm(range(num_samples), desc="Evaluating pitch control"):
            audio, original_midi = dataset[i]
            audio = audio.unsqueeze(0).unsqueeze(0).to(device)
            original_midi = original_midi.to(device)
            
            # Get latent codes
            orig_z = original_model.encode(audio)
            dis_z = disentangled_model.encode(audio)
            
            # Test pitch change (shift by +7 semitones)
            new_pitch = (original_midi + 7) % 88  # Wrap around if needed
            
            # Original model reconstruction with new pitch
            # (Note: original model doesn't have proper pitch conditioning)
            orig_recon_new = original_model.decode(orig_z)
            
            # Disentangled model with new pitch
            dis_recon_new = disentangled_model.decode(dis_z, new_pitch)
            
            # Calculate pitch accuracy (crude F0 estimation)
            def estimate_pitch(audio):
                audio_np = audio.squeeze().cpu().numpy()
                fft = np.fft.rfft(audio_np)
                frequencies = np.fft.rfftfreq(len(audio_np), 1/16000)
                magnitude = np.abs(fft)
                peak_freq = frequencies[np.argmax(magnitude[1:]) + 1]  # Skip DC
                return peak_freq
            
            # Calculate expected frequency for original and new pitch
            expected_orig_freq = 440 * (2 ** ((original_midi.item() + 21 - 69) / 12))
            expected_new_freq = 440 * (2 ** ((new_pitch.item() + 21 - 69) / 12))
            
            # Estimate actual frequencies
            orig_freq = estimate_pitch(orig_recon_new)
            dis_freq = estimate_pitch(dis_recon_new)
            
            # Calculate accuracy (within ±5% of expected frequency)
            orig_acc = 1 if abs(orig_freq - expected_orig_freq)/expected_orig_freq < 0.05 else 0
            dis_acc = 1 if abs(dis_freq - expected_new_freq)/expected_new_freq < 0.05 else 0
            
            pitch_accuracies['original'].append(orig_acc)
            pitch_accuracies['disentangled'].append(dis_acc)
            
            # Plot first sample
            if i == 0:
                plt.figure(figsize=(12, 8))
                
                plt.subplot(2, 1, 1)
                plt.plot(orig_recon_new.squeeze().cpu().numpy())
                plt.title(f"Original Model - Expected: {expected_orig_freq:.1f}Hz, Actual: {orig_freq:.1f}Hz")
                
                plt.subplot(2, 1, 2)
                plt.plot(dis_recon_new.squeeze().cpu().numpy())
                plt.title(f"Disentangled Model - Expected: {expected_new_freq:.1f}Hz, Actual: {dis_freq:.1f}Hz")
                
                plt.tight_layout()
                plt.show()
                
                print("Original Model (no pitch control):")
                display(Audio(orig_recon_new.squeeze().cpu().numpy(), rate=16000))
                print("Disentangled Model (with pitch control):")
                display(Audio(dis_recon_new.squeeze().cpu().numpy(), rate=16000))
    
    results['pitch_accuracy'] = {
        'original': np.mean(pitch_accuracies['original']),
        'disentangled': np.mean(pitch_accuracies['disentangled'])
    }
    print(f"\nPitch Control Accuracy:")
    print(f"Original: {results['pitch_accuracy']['original']*100:.1f}%")
    print(f"Disentangled: {results['pitch_accuracy']['disentangled']*100:.1f}%")
    
    # 3. Timbre Preservation Across Pitches
    print("\n=== Timbre Preservation ===")
    timbre_scores = []
    with torch.no_grad():
        for i in tqdm(range(min(num_samples, 3)), desc="Evaluating timbre preservation"):
            audio, original_midi = dataset[i]
            audio = audio.unsqueeze(0).unsqueeze(0).to(device)
            original_midi = original_midi.to(device)
            
            # Get latent code from disentangled model
            z = disentangled_model.encode(audio)
            
            # Generate at original pitch and +12 semitones
            recon_orig = disentangled_model.decode(z, original_midi)
            recon_high = disentangled_model.decode(z, (original_midi + 12) % 88)
            
            # Calculate spectral similarity (crude measure)
            def spectral_features(audio):
                audio_np = audio.squeeze().cpu().numpy()
                fft = np.fft.rfft(audio_np)
                magnitude = np.abs(fft)[1:50]  # First 50 bins (skip DC)
                return magnitude / np.sum(magnitude)
            
            feat_orig = spectral_features(recon_orig)
            feat_high = spectral_features(recon_high)
            
            # Cosine similarity
            similarity = np.dot(feat_orig, feat_high) / (np.linalg.norm(feat_orig) * np.linalg.norm(feat_high))
            timbre_scores.append(similarity)
            
            # Plot first sample
            if i == 0:
                plt.figure(figsize=(12, 6))
                
                plt.subplot(1, 2, 1)
                plt.bar(range(len(feat_orig)), feat_orig)
                plt.title(f"Original Pitch {original_midi.item() + 21}")
                
                plt.subplot(1, 2, 2)
                plt.bar(range(len(feat_high)), feat_high)
                plt.title(f"Higher Pitch {(original_midi.item() + 12) % 88 + 21}")
                
                plt.suptitle(f"Spectral Similarity: {similarity:.3f}")
                plt.tight_layout()
                plt.show()
                
                print("Original Pitch:")
                display(Audio(recon_orig.squeeze().cpu().numpy(), rate=16000))
                print("Higher Pitch:")
                display(Audio(recon_high.squeeze().cpu().numpy(), rate=16000))
    
    results['timbre_similarity'] = np.mean(timbre_scores)
    print(f"\nAverage Timbre Similarity Across Pitches: {results['timbre_similarity']:.3f}")
    
    # 4. Latent Space Analysis
    print("\n=== Latent Space Analysis ===")
    with torch.no_grad():
        # Collect embeddings for different pitches
        all_embeddings = {'original': [], 'disentangled': []}
        all_pitches = []
        
        # Get multiple samples with different pitches
        for i in range(10):
            audio, midi = dataset[np.random.randint(len(dataset))]
            audio = audio.unsqueeze(0).unsqueeze(0).to(device)
            midi = midi.to(device)
            
            # Get embeddings
            orig_emb = original_model.encode(audio)
            dis_emb = disentangled_model.encode(audio)
            
            # Average over time dimension
            all_embeddings['original'].append(orig_emb.mean(-1).squeeze().cpu().numpy())
            all_embeddings['disentangled'].append(dis_emb.mean(-1).squeeze().cpu().numpy())
            all_pitches.append(midi.item())
        
        # Convert to numpy arrays
        orig_embeddings = np.array(all_embeddings['original'])
        dis_embeddings = np.array(all_embeddings['disentangled'])
        pitches = np.array(all_pitches)
        
        # Calculate pitch correlation for each dimension
        def calculate_correlations(embeddings, pitches):
            correlations = []
            for dim in range(embeddings.shape[1]):
                corr = np.corrcoef(embeddings[:, dim], pitches)[0, 1]
                correlations.append(abs(corr))
            return np.mean(correlations)
        
        orig_corr = calculate_correlations(orig_embeddings, pitches)
        dis_corr = calculate_correlations(dis_embeddings, pitches)
        
        results['pitch_correlation'] = {
            'original': orig_corr,
            'disentangled': dis_corr
        }
        
        print(f"\nPitch Information in Latent Space:")
        print(f"Original Model: {orig_corr:.3f}")
        print(f"Disentangled Model: {dis_corr:.3f}")
        
        # Plot first 2 dimensions
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.scatter(orig_embeddings[:, 0], orig_embeddings[:, 1], c=pitches)
        plt.colorbar(label='MIDI Pitch')
        plt.title("Original Model Latent Space")
        
        plt.subplot(1, 2, 2)
        plt.scatter(dis_embeddings[:, 0], dis_embeddings[:, 1], c=pitches)
        plt.colorbar(label='MIDI Pitch')
        plt.title("Disentangled Model Latent Space")
        
        plt.tight_layout()
        plt.show()
    
    return results