import torch
import numpy as np
from torch.utils.data import Dataset

class SyntheticNotesDataset(Dataset):
    def __init__(self, num_samples=5000, sample_rate=16000, duration=1.0):
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.duration = duration
        self.num_samples_per_note = int(sample_rate * duration)
        
        # MIDI note range (A0 to C8)
        self.min_midi = 21
        self.max_midi = 108
        
        # For timbre variation
        self.harmonic_factors = [
            lambda n: 1.0/n,              # Standard harmonic series
            lambda n: 1.0/(n**1.5),       # Brighter sound
            lambda n: 1.0/(n**2),         # Darker sound
            lambda n: 1.0/n if n%2==1 else 0,  # Odd harmonics only
        ]
        
    def midi_to_freq(self, midi_note):
        """Convert MIDI note number to frequency"""
        return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))
    
    def generate_note(self, idx):
        """Generate a synthetic musical note"""
        # Random pitch
        midi_note = np.random.randint(self.min_midi, self.max_midi + 1)
        freq = self.midi_to_freq(midi_note)
        
        # Random timbre
        harmonic_profile = np.random.choice(self.harmonic_factors)
        
        # Generate time array
        t = np.linspace(0, self.duration, self.num_samples_per_note, endpoint=False)
        
        # Generate harmonics (up to 10th harmonic)
        signal = np.zeros_like(t)
        for n in range(1, 11):
            harmonic_freq = freq * n
            amplitude = harmonic_profile(n)
            if amplitude > 0:
                phase = np.random.uniform(0, 2*np.pi)  # Random phase for each harmonic
                signal += amplitude * np.sin(2 * np.pi * harmonic_freq * t + phase)
        
        # Apply ADSR envelope
        attack = np.random.uniform(0.01, 0.1)
        decay = np.random.uniform(0.05, 0.3)
        sustain_level = np.random.uniform(0.3, 0.8)
        release = np.random.uniform(0.1, 0.5)
        
        # Create envelope
        total_samples = len(t)
        attack_samples = int(attack * total_samples)
        decay_samples = int(decay * total_samples)
        release_samples = int(release * total_samples)
        sustain_samples = total_samples - attack_samples - decay_samples - release_samples
        
        if sustain_samples < 0:
            # If release starts before sustain, adjust
            sustain_samples = 0
            release_samples = total_samples - attack_samples - decay_samples
        
        envelope = np.concatenate([
            np.linspace(0, 1, attack_samples),  # Attack
            np.linspace(1, sustain_level, decay_samples),  # Decay
            np.ones(sustain_samples) * sustain_level,  # Sustain
            np.linspace(sustain_level, 0, release_samples)  # Release
        ])
        
        # Apply envelope
        signal = signal * envelope[:len(signal)]
        
        # Add some noise to attack
        if np.random.rand() > 0.3:  # 70% chance of adding attack noise
            noise_duration = np.random.uniform(0.001, 0.02)
            noise_samples = int(noise_duration * self.sample_rate)
            if noise_samples > 0:
                noise = np.random.uniform(-0.5, 0.5, noise_samples)
                signal[:noise_samples] += noise * np.linspace(1, 0, noise_samples)
        
        # Normalize
        signal = signal / (np.max(np.abs(signal)) + 1e-6)
        
        return signal, midi_note
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        audio, midi_note = self.generate_note(idx)
        
        # Convert to torch tensors
        audio_tensor = torch.FloatTensor(audio)
        midi_tensor = torch.LongTensor([midi_note - self.min_midi])  # Make zero-based
        
        return audio_tensor, midi_tensor