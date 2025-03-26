import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Enhanced WaveNet Autoencoder with Disentangled Representations
class DisentangledWaveNetAE(nn.Module):
    def __init__(self, input_length=8000, latent_dim=128, num_dilated_layers=5, 
                 bottleneck_dim=32, num_pitches=88):
        super(DisentangledWaveNetAE, self).__init__()
        self.input_length = input_length
        self.latent_dim = latent_dim
        self.bottleneck_dim = bottleneck_dim
        self.num_pitches = num_pitches
        
        self.encoder_conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        
        self.encoder_dilated_convs = nn.ModuleList()
        for i in range(num_dilated_layers):
            dilation = 2 ** i
            self.encoder_dilated_convs.append(
                nn.Conv1d(64, 64, kernel_size=3, padding=dilation, dilation=dilation)
            )
        
        # Information bottleneck layer
        self.bottleneck = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, bottleneck_dim, kernel_size=3, stride=2, padding=1),
            nn.Tanh()
        )
        
        # Pitch discriminator
        self.pitch_discriminator = nn.Sequential(
            nn.Conv1d(bottleneck_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, num_pitches)
        )
        
        # Decoder now takes both bottleneck and pitch conditioning
        self.pitch_embedding = nn.Embedding(num_pitches, 16)
        
        self.decoder_initial = nn.Sequential(
            nn.Conv1d(bottleneck_dim + 16, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
        )
        
        self.decoder_dilated_convs = nn.ModuleList()
        for i in reversed(range(num_dilated_layers)):
            dilation = 2 ** i
            self.decoder_dilated_convs.append(
                nn.Conv1d(64, 64, kernel_size=3, padding=dilation, dilation=dilation)
            )
        
        self.decoder_final = nn.Conv1d(64, 1, kernel_size=3, padding=1)
    
    def encode(self, x):
        x = F.relu(self.encoder_conv1(x))
        for conv in self.encoder_dilated_convs:
            x = F.relu(conv(x))
        z = self.bottleneck(x)
        return z
    
    def decode(self, z, pitch):
        # Embed pitch and expand to match z's temporal dimensions
        pitch_emb = self.pitch_embedding(pitch)
        pitch_emb = pitch_emb.unsqueeze(-1).expand(-1, -1, z.shape[-1])
        
        # Concatenate along channel dimension
        z = torch.cat([z, pitch_emb], dim=1)
        
        z = self.decoder_initial(z)
        for conv in self.decoder_dilated_convs:
            z = F.relu(conv(z))
        z = torch.tanh(self.decoder_final(z))
        return z
    
    def forward(self, x, pitch):
        z = self.encode(x)
        x_recon = self.decode(z, pitch)
        return x_recon
    
    def predict_pitch(self, z):
        return self.pitch_discriminator(z)

# Simplified WaveNet Autoencoder
class WaveNetAutoencoder(nn.Module):
    def __init__(self, input_length=8000, latent_dim=128, num_dilated_layers=5):
        super(WaveNetAutoencoder, self).__init__()
        self.input_length = input_length
        self.latent_dim = latent_dim
        
        # Encoder: A simplified WaveNet-style encoder
        self.encoder_conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        
        # Dilated convolutions for encoder
        self.encoder_dilated_convs = nn.ModuleList()
        for i in range(num_dilated_layers):
            dilation = 2 ** i
            self.encoder_dilated_convs.append(
                nn.Conv1d(64, 64, kernel_size=3, padding=dilation, dilation=dilation)
            )
        
        self.encoder_final = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, latent_dim, kernel_size=3, stride=2, padding=1),
            nn.Tanh()  # Normalize to [-1, 1]
        )
        
        # Calculate the compressed length after convolutions
        self.compressed_length = input_length // 8
        
        # Decoder: A simplified WaveNet-style decoder
        self.decoder_initial = nn.Sequential(
            nn.Conv1d(latent_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
        )
        
        # Dilated convolutions for decoder
        self.decoder_dilated_convs = nn.ModuleList()
        for i in reversed(range(num_dilated_layers)):
            dilation = 2 ** i
            self.decoder_dilated_convs.append(
                nn.Conv1d(64, 64, kernel_size=3, padding=dilation, dilation=dilation)
            )
        
        self.decoder_final = nn.Conv1d(64, 1, kernel_size=3, padding=1)
    
    def encode(self, x):
        # x shape: (batch_size, 1, input_length)
        x = F.relu(self.encoder_conv1(x))
        
        for conv in self.encoder_dilated_convs:
            x = F.relu(conv(x))
        
        x = self.encoder_final(x)
        return x  # (batch_size, latent_dim, compressed_length)
    
    def decode(self, z):
        # z shape: (batch_size, latent_dim, compressed_length)
        z = self.decoder_initial(z)
        
        for conv in self.decoder_dilated_convs:
            z = F.relu(conv(z))
        
        z = torch.tanh(self.decoder_final(z))  # Normalize to [-1, 1]
        return z
    
    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon