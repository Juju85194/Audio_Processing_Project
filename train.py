import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

# Training loop
def train_model(model, dataloader, optimizer, criterion, num_epochs=20, device="cuda" if torch.cuda.is_available() else "cpu"):
    model.train()
    
    # Initialize progress bar for epochs
    epoch_pbar = tqdm(range(num_epochs), desc="Training Progress")
    
    for epoch in epoch_pbar:
        running_loss = 0.0
        
        for i, (audio, _) in enumerate(dataloader):
            # Prepare input
            audio = audio.unsqueeze(1).to(device)  # Add channel dim
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(audio)
            
            # Compute loss
            loss = criterion(outputs, audio)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Update batch progress bar
            epoch_pbar.set_postfix({"Batch Loss": f"{loss.item():.6f}"})
        
        # Calculate average epoch loss
        epoch_loss = running_loss / len(dataloader)
        
        # Print epoch summary (optional)
        print(f"\nEpoch [{epoch+1}/{num_epochs}], Average Loss: {epoch_loss:.6f}")
    
    print('Training complete!')

# Training setup with adversarial component
def train_disentangled(model, dataloader, optimizer, criterion, num_epochs=20, adv_weight=1e-4, device="cuda" if torch.cuda.is_available() else "cpu"):
    model.train()
    
    # Loss functions
    recon_criterion = nn.MSELoss()
    pitch_criterion = nn.CrossEntropyLoss()
    
    # Optimizers
    main_optimizer = torch.optim.Adam([
        {'params': model.encoder_conv1.parameters()},
        {'params': model.encoder_dilated_convs.parameters()},
        {'params': model.bottleneck.parameters()},
        {'params': model.pitch_embedding.parameters()},
        {'params': model.decoder_initial.parameters()},
        {'params': model.decoder_dilated_convs.parameters()},
        {'params': model.decoder_final.parameters()}
    ], lr=0.001)
    
    disc_optimizer = torch.optim.Adam(model.pitch_discriminator.parameters(), lr=0.0001)
    
    # Progress bars
    epoch_pbar = tqdm(range(num_epochs), desc="Training Progress")
    
    for epoch in epoch_pbar:
        running_recon_loss = 0.0
        running_pitch_loss = 0.0
        running_disc_loss = 0.0
        
        for audio, midi in dataloader:
            audio = audio.unsqueeze(1).to(device)
            midi = midi.squeeze().to(device)
            
            # --- MAIN OPTIMIZATION (reconstruction + adversarial) ---
            main_optimizer.zero_grad()
            
            # Forward pass
            z = model.encode(audio)
            pitch_pred = model.predict_pitch(z)
            audio_recon = model.decode(z, midi)
            
            # Reconstruction loss
            recon_loss = recon_criterion(audio_recon, audio)
            
            # Adversarial loss - try to fool discriminator
            pitch_loss = pitch_criterion(pitch_pred, midi)
            
            # Total loss with adversarial component
            # We want to MINIMIZE pitch classification accuracy
            total_loss = recon_loss - adv_weight * pitch_loss  
            
            total_loss.backward()
            main_optimizer.step()
            
            # --- DISCRIMINATOR OPTIMIZATION ---
            disc_optimizer.zero_grad()
            
            # Forward pass with detached z
            with torch.no_grad():
                z = model.encode(audio)
            
            pitch_pred = model.predict_pitch(z)
            disc_loss = pitch_criterion(pitch_pred, midi)
            
            disc_loss.backward()
            disc_optimizer.step()
            
            # Update running losses
            running_recon_loss += recon_loss.item()
            running_pitch_loss += pitch_loss.item()
            running_disc_loss += disc_loss.item()
            
            epoch_pbar.set_postfix({
                "Recon Loss": f"{recon_loss.item():.4f}",
                "Pitch Acc": f"{(pitch_pred.argmax(1) == midi).float().mean().item():.2f}"
            })
        
        # Epoch statistics
        avg_recon = running_recon_loss / len(dataloader)
        avg_pitch = running_pitch_loss / len(dataloader)
        avg_disc = running_disc_loss / len(dataloader)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Reconstruction Loss: {avg_recon:.6f}")
        print(f"Pitch Prediction Loss: {avg_pitch:.6f}")
        print(f"Discriminator Loss: {avg_disc:.6f}")