import torch
import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training loop
def train_model(model, dataloader, num_epochs=20):
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