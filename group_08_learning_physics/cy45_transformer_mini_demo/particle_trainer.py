import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from particle_models import ParticleDataset, PhysicsInformedParticleTransformer

def physics_informed_loss(pred, target, gravity_weight=0.01, bounds=(0.1, 0.9), dt=0.01):
    """Combined loss function with physics constraints"""
    
    # Extract positions and velocities
    pred_pos, pred_vel = pred[:, :, :2], pred[:, :, 2:]
    true_pos, true_vel = target[:, :, :2], target[:, :, 2:]
    
    # Standard MSE losses
    pos_loss = F.mse_loss(pred_pos, true_pos)
    vel_loss = F.mse_loss(pred_vel, true_vel)
    
    # Softer boundary violation penalty
    min_bound, max_bound = bounds
    margin = 0.02
    
    # Only penalize significant violations
    violation_min = torch.relu(min_bound - pred_pos)
    violation_max = torch.relu(pred_pos - max_bound)
    boundary_loss = (violation_min + violation_max).mean()
    
    # Combine losses
    total_loss = pos_loss * 100 + vel_loss * 1 + boundary_loss * 0.01
    
    return {
        'total_loss': total_loss,
        'pos_loss': pos_loss,
        'vel_loss': vel_loss,
        'boundary_loss': boundary_loss
    }

def custom_collate_fn(batch):
    """Custom collate function to handle variable number of particles"""
    # Find the maximum number of particles in this batch
    max_particles = max(x.shape[1] for x, y in batch)
    
    batch_x = []
    batch_y = []
    
    for x, y in batch:
        seq_len, num_particles, features = x.shape
        
        # Pad with zeros if needed
        if num_particles < max_particles:
            pad_size = max_particles - num_particles
            x_pad = torch.zeros(seq_len, pad_size, features)
            y_pad = torch.zeros(seq_len, pad_size, features)
            
            x = torch.cat([x, x_pad], dim=1)
            y = torch.cat([y, y_pad], dim=1)
        
        batch_x.append(x)
        batch_y.append(y)
    
    return torch.stack(batch_x), torch.stack(batch_y)

def train_model(model, train_loader, val_loader, num_epochs=100, lr=1e-3, device='cuda'):
    """Training loop for the particle transformer"""
    
    # Separate optimizers for different parts
    main_params = [p for name, p in model.named_parameters() if 'gravity' not in name]
    gravity_params = [p for name, p in model.named_parameters() if 'gravity' in name]
    
    optimizer = torch.optim.Adam([
        {'params': main_params, 'lr': lr},
        {'params': gravity_params, 'lr': lr * 0.1}  # Slower learning for gravity
    ], weight_decay=1e-5)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
    
    model.to(device)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0
        num_batches = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Handle sequence dimension - process each timestep
            batch_size, seq_len, num_particles, features = batch_x.shape
            
            optimizer.zero_grad()
            total_loss = 0
            
            # Process each timestep in the sequence
            for t in range(seq_len):
                pred = model(batch_x[:, t])  # (batch, particles, 4)
                target = batch_y[:, t]       # (batch, particles, 4)
                
                loss_dict = physics_informed_loss(pred, target)
                total_loss += loss_dict['total_loss']
            
            total_loss = total_loss / seq_len
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_train_loss += total_loss.item()
            num_batches += 1
        
        avg_train_loss = epoch_train_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        epoch_val_loss = 0
        num_val_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                batch_size, seq_len, num_particles, features = batch_x.shape
                
                total_val_loss = 0
                for t in range(seq_len):
                    pred = model(batch_x[:, t])
                    target = batch_y[:, t]
                    
                    loss_dict = physics_informed_loss(pred, target)
                    total_val_loss += loss_dict['total_loss']
                
                total_val_loss = total_val_loss / seq_len
                epoch_val_loss += total_val_loss.item()
                num_val_batches += 1
        
        avg_val_loss = epoch_val_loss / num_val_batches
        val_losses.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        if epoch % 5 == 0:
            print(f'Epoch {epoch:3d}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}, Gravity = [0.0, {model.gravity.item():.4f}]')
    
    return train_losses, val_losses

def save_training_curves(train_losses, val_losses, save_path='training_curves.png'):
    """Save training and validation loss curves"""
    plt.switch_backend('Agg')
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Progress')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to '{save_path}'")
    plt.close()

# Main training script
if __name__ == "__main__":
    # Set matplotlib backend for headless operation
    plt.switch_backend('Agg')
    
    print("=== Particle Physics Transformer Training ===")
    
    # Load dataset
    print("Loading dataset...")
    dataset = ParticleDataset('sample_data/water_drop/single_trajectory.npz', sequence_length=15)
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=custom_collate_fn)
    
    # Initialize model
    print("\nInitializing model...")
    model = PhysicsInformedParticleTransformer(
        d_model=128,
        n_heads=8,
        n_layers=3,
        dropout=0.01,
        gravity=0.005,  # Small gravity
        bounds=(0.1, 0.9),
        dt=0.05
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on device: {device}")
    
    print("\n=== Starting Training ===")
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, 
        num_epochs=50, lr=1e-3, device=device
    )
    
    # Save training curves
    save_training_curves(train_losses, val_losses)
    
    # Save model
    torch.save(model.state_dict(), 'particle_transformer.pth')
    print("Model saved as 'particle_transformer.pth'")
    
    # Save training metadata
    training_info = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'final_gravity': [0.0, model.gravity.item()],
        'model_config': {
            'd_model': 128,
            'n_heads': 8,
            'n_layers': 3,
            'dropout': 0.01,
            'bounds': (0.1, 0.9),
            'dt': 0.05
        }
    }
    torch.save(training_info, 'training_info.pth')
    print("Training info saved as 'training_info.pth'")
    
    print("\n=== Training Complete ===")
    print(f"Final training loss: {train_losses[-1]:.6f}")
    print(f"Final validation loss: {val_losses[-1]:.6f}")
    print(f"Learned gravity: [0.0, {model.gravity.item():.4f}]")
    print("\nTo evaluate the model, run: python particle_evaluator.py")