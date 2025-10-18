import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset

class ParticleDataset(Dataset):
    """Dataset for particle simulation trajectories"""
    
    def __init__(self, npz_file, sequence_length=20):
        """
        Args:
            npz_file: Path to the .npz file containing simulation data
            sequence_length: Number of consecutive frames to use for training
        """
        self.data = np.load(npz_file, allow_pickle=True)
        self.sequence_length = sequence_length
        
        # Extract trajectories
        self.trajectories = []
        trajectory_idx = 0
        
        while f'simulation_trajectory_{trajectory_idx}' in self.data:
            traj_data = self.data[f'simulation_trajectory_{trajectory_idx}']
            positions = traj_data[0]  # Shape: (timesteps, particles, 2)
            materials = traj_data[1]  # Shape: (particles,)
            
            # Compute velocities using discrete differentiation
            velocities = np.zeros_like(positions)
            velocities[1:] = positions[1:] - positions[:-1]  # v_t = pos_t - pos_{t-1}
            velocities[0] = velocities[1]  # Copy first velocity
            
            # Combine positions and velocities: [pos_x, pos_y, vel_x, vel_y]
            state = np.concatenate([positions, velocities], axis=-1)  # (timesteps, particles, 4)
            
            self.trajectories.append(state)
            trajectory_idx += 1
        
        print(f"Loaded {len(self.trajectories)} trajectories")
        if len(self.trajectories) > 0:
            print(f"First trajectory shape: {self.trajectories[0].shape}")
    
    def __len__(self):
        total_samples = 0
        for traj in self.trajectories:
            total_samples += max(0, traj.shape[0] - self.sequence_length)
        return total_samples
    
    def __getitem__(self, idx):
        # Find which trajectory and timestep this index corresponds to
        current_idx = idx
        for traj in self.trajectories:
            max_start = max(0, traj.shape[0] - self.sequence_length)
            if current_idx < max_start:
                start_t = current_idx
                sequence = traj[start_t:start_t + self.sequence_length]
                
                # Return input (first seq_len-1 frames) and target (last seq_len-1 frames)
                x = torch.FloatTensor(sequence[:-1])  # (seq_len-1, particles, 4)
                y = torch.FloatTensor(sequence[1:])   # (seq_len-1, particles, 4)
                
                return x, y
            current_idx -= max_start
        
        raise IndexError("Dataset index out of range")
    
    def get_full_trajectory(self, traj_idx):
        """Get a complete trajectory for rollout evaluation"""
        if traj_idx >= len(self.trajectories):
            raise IndexError(f"Trajectory index {traj_idx} out of range")
        return torch.FloatTensor(self.trajectories[traj_idx])

def add_boundary_features(x, bounds=(0.1, 0.9)):
    """Add distance-to-boundary features"""
    positions = x[:, :, :2]  # Extract positions (batch, particles, 2)
    min_bound, max_bound = bounds
    
    # Distance to each boundary
    dist_to_min = positions - min_bound  # Distance to left/bottom
    dist_to_max = max_bound - positions  # Distance to right/top
    
    boundary_features = torch.cat([dist_to_min, dist_to_max], dim=-1)
    return torch.cat([x, boundary_features], dim=-1)  # (batch, particles, 8)

def apply_boundary_constraints(positions, velocities, bounds=(0.1, 0.9), damping=0.8):
    """Apply hard boundary constraints with reflection and damping"""
    min_bound, max_bound = bounds
    
    # Clamp positions to boundaries
    positions_clamped = torch.clamp(positions, min_bound, max_bound)
    
    # Detect boundary collisions
    hit_min = positions <= min_bound
    hit_max = positions >= max_bound
    
    # Reflect and damp velocities at boundaries
    velocities_new = velocities.clone()
    velocities_new = torch.where(hit_min, torch.abs(velocities) * damping, velocities_new)
    velocities_new = torch.where(hit_max, -torch.abs(velocities) * damping, velocities_new)
    
    return positions_clamped, velocities_new

class PhysicsInformedParticleTransformer(nn.Module):
    """Transformer-based model for particle dynamics prediction"""
    
    def __init__(self, d_model=128, n_heads=8, n_layers=3, dropout=0.1, 
                 gravity=0.005, bounds=(0.1, 0.9), dt=0.01):  # Small gravity
        super().__init__()
        
        # Model parameters
        self.d_model = d_model
        self.bounds = bounds
        self.dt = dt
        
        # Input: [pos_x, pos_y, vel_x, vel_y, dist_to_min_x, dist_to_min_y, dist_to_max_x, dist_to_max_y]
        self.input_dim = 8
        
        # Small learnable gravity (can be fine-tuned during training)
        self.gravity = nn.Parameter(torch.tensor(gravity))
        # Register zero as a buffer so it moves with the model to different devices
        self.register_buffer('zero', torch.tensor(0.0))
        
        # Network layers
        self.input_projection = nn.Linear(self.input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1000, d_model))  # Max 1000 particles
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output projection to [pos_x, pos_y, vel_x, vel_y]
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 4)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, num_particles, 4)
               Contains [pos_x, pos_y, vel_x, vel_y] for each particle

        Returns:
            Predicted next state of shape (batch_size, num_particles, 4)
        """
        batch_size, num_particles, _ = x.shape
        
        # Add boundary features
        x_augmented = add_boundary_features(x, self.bounds)  # (batch, particles, 8)
        
        # Project to model dimension
        embeddings = self.input_projection(x_augmented)  # (batch, particles, d_model)
        
        # Add positional embeddings (particle-specific, not spatial)
        embeddings = embeddings + self.pos_embedding[:num_particles].unsqueeze(0)
        embeddings = self.layer_norm(embeddings)
        
        # Apply transformer layers
        transformed = self.transformer(embeddings)  # (batch, particles, d_model)
        
        # Project to output space
        output = self.output_projection(transformed)  # (batch, particles, 4)
        
        # Split current state
        current_pos = x[:, :, :2]
        current_vel = x[:, :, 2:]
        
        # Split into positions and velocities
        pred_pos = output[:, :, :2]
        pred_vel = output[:, :, 2:]
        
        # Apply physics: add gravity to velocity predictions
        gravity_y = torch.stack([self.zero, self.gravity])
        gravity_effect = gravity_y.unsqueeze(0).unsqueeze(0) * self.dt
        pred_vel = pred_vel + gravity_effect
        pred_pos = pred_pos + pred_vel * self.dt
        
        # Apply soft boundary constraints
        pred_pos, pred_vel = apply_boundary_constraints(pred_pos, pred_vel, self.bounds, damping=0.9)
        
        # Combine and return
        return torch.cat([pred_pos, pred_vel], dim=-1)
    
    def multi_step_rollout(self, initial_state, num_steps, device='cpu'):
        """
        Perform multi-step rollout prediction
        
        Args:
            initial_state: Initial particle state (1, num_particles, 4)
            num_steps: Number of steps to predict
            device: Device to run on

        Returns:
            rollout: Predicted trajectory (num_steps+1, num_particles, 4)
        """
        self.eval()
        rollout = [initial_state.clone()]
        current_state = initial_state.to(device)
        
        with torch.no_grad():
            for step in range(num_steps):
                next_state = self.forward(current_state)
                rollout.append(next_state.cpu())
                current_state = next_state
        
        return torch.cat(rollout, dim=0)  # (num_steps+1, num_particles, 4)