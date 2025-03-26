import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

# Training loop
def train_model(model, train_dataloader, val_dataloader, optimizer, criterion, num_epochs=20, device="cuda" if torch.cuda.is_available() else "cpu"):
    model.train()
    
    # Initialize progress bar for epochs
    epoch_pbar = tqdm(range(num_epochs), desc="Training Progress")
    
    for epoch in epoch_pbar:
        running_loss = 0.0
        val_loss = 0.0
        
        # Training loop
        for i, (audio, _) in enumerate(train_dataloader):
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
        epoch_loss = running_loss / len(train_dataloader)
        
        # Validation loop
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation for validation
            for audio, _ in val_dataloader:
                audio = audio.unsqueeze(1).to(device)
                outputs = model(audio)
                loss = criterion(outputs, audio)
                val_loss += loss.item()
        model.train()  # Set model back to training mode

        val_loss /= len(val_dataloader)
        
        # Print epoch summary (optional)
        print(f"\nEpoch [{epoch+1}/{num_epochs}], Average Loss: {epoch_loss:.6f}, Validation Loss: {val_loss:.6f}")
    
    print('Training complete!')

# Training setup with adversarial component
def train_disentangled(model, train_dataloader, val_dataloader, optimizer, criterion, num_epochs=20, adv_weight=0.01, decay=0.01, device="cuda" if torch.cuda.is_available() else "cpu"):
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

        val_recon_loss = 0.0
        val_pitch_loss = 0.0
        val_disc_loss = 0.0
        
        for audio, midi in train_dataloader:
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
            adv_weight = adv_weight * (1 - decay)
            
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
                "Pitch Acc": f"{(pitch_pred.argmax(1) == midi).float().mean().item():.2f}",
                "Adv weight": f"{adv_weight:.4f}"
            })

        # --- VALIDATION LOOP ---
        model.eval()
        with torch.no_grad():
            for audio, midi in val_dataloader:
                audio = audio.unsqueeze(1).to(device)
                midi = midi.squeeze().to(device)

                z = model.encode(audio)
                pitch_pred = model.predict_pitch(z)
                audio_recon = model.decode(z, midi)

                recon_loss = recon_criterion(audio_recon, audio)
                pitch_loss = pitch_criterion(pitch_pred, midi)
                disc_loss = pitch_criterion(pitch_pred, midi)  # Discriminator loss calculation

                val_recon_loss += recon_loss.item()
                val_pitch_loss += pitch_loss.item()
                val_disc_loss += disc_loss.item()

        model.train()  # Set back to training mode
        
        # Epoch statistics
        avg_recon = running_recon_loss / len(train_dataloader)
        avg_pitch = running_pitch_loss / len(train_dataloader)
        avg_disc = running_disc_loss / len(train_dataloader)

        avg_val_recon = val_recon_loss / len(val_dataloader)
        avg_val_pitch = val_pitch_loss / len(val_dataloader)
        avg_val_disc = val_disc_loss / len(val_dataloader)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Reconstruction Loss: {avg_recon:.6f}, Validation Reconstruction Loss: {avg_val_recon:.6f}")
        print(f"Pitch Prediction Loss: {avg_pitch:.6f}, Validation Pitch Prediction Loss: {avg_val_pitch:.6f}")
        print(f"Discriminator Loss: {avg_disc:.6f}, Validation Discriminator Loss: {avg_val_disc:.6f}")