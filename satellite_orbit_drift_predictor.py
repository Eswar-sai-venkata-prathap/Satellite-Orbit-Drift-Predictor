import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import (
    confusion_matrix, 
    accuracy_score, 
    roc_auc_score, 
    ConfusionMatrixDisplay
)
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')


np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


EARTH_RADIUS_KM = 6371.0  
MU_EARTH = 398600.4418    
J2 = 1.08263e-3          


NUM_SEQUENCES = 2000
INPUT_TIMESTEPS = 100      
PREDICTION_HORIZON = 50   
TOTAL_TIMESTEPS = INPUT_TIMESTEPS + PREDICTION_HORIZON
NUM_FEATURES = 6          

INITIAL_SEMI_MAJOR_AXIS = EARTH_RADIUS_KM + 500 
INITIAL_ECCENTRICITY = 0.001                     
INITIAL_INCLINATION = np.radians(51.6)            
INITIAL_RAAN = np.radians(0.0)                    
INITIAL_ARG_PERIGEE = np.radians(0.0)             
INITIAL_MEAN_ANOMALY = np.radians(0.0)            


DANGER_ZONE_A = EARTH_RADIUS_KM + 450 
HIGH_RISK_A_THRESHOLD = EARTH_RADIUS_KM + 480  
DANGER_ZONE_E = 0.01  

TRAIN_SIZE = 1400
VAL_SIZE = 300
TEST_SIZE = 300
BATCH_SIZE = 64
HIDDEN_DIM = 128
NUM_LAYERS = 2
LEARNING_RATE = 0.001
NUM_EPOCHS = 50


def compute_orbital_period(semi_major_axis):
    """
    Compute orbital period using Kepler's third law.
    T = 2π * sqrt(a³/μ)
    """
    return 2 * np.pi * np.sqrt(semi_major_axis**3 / MU_EARTH)


def compute_j2_perturbations(a, e, i, dt):
    """
    Compute J2 perturbation effects on RAAN and argument of perigee.
    
    J2 causes:
    - Nodal regression (RAAN drift)
    - Apsidal precession (argument of perigee drift)
    """
    n = np.sqrt(MU_EARTH / a**3)  # Mean motion
    p = a * (1 - e**2)            # Semi-latus rectum
    
    # Rate of RAAN regression (rad/s)
    raan_rate = -1.5 * n * J2 * (EARTH_RADIUS_KM / p)**2 * np.cos(i)
    
    # Rate of argument of perigee precession (rad/s)
    omega_rate = 0.75 * n * J2 * (EARTH_RADIUS_KM / p)**2 * (5 * np.cos(i)**2 - 1)
    
    return raan_rate * dt, omega_rate * dt


def compute_atmospheric_drag_decay(a, e, timestep, base_decay_rate):
    """
    Compute semi-major axis decay due to atmospheric drag.
    
    Atmospheric density increases exponentially as altitude decreases,
    causing accelerated orbital decay at lower altitudes.
    """
    altitude = a - EARTH_RADIUS_KM
    
    # Exponential atmosphere model: density increases as altitude decreases
    # Reference altitude: 500 km
    reference_altitude = 500.0
    scale_height = 60.0  # Atmospheric scale height in km
    
    # Density ratio compared to reference altitude
    density_ratio = np.exp((reference_altitude - altitude) / scale_height)
    
    # Decay rate increases with density
    decay = base_decay_rate * density_ratio * (1 + 0.5 * e)
    
    return decay


def generate_single_sequence(sequence_id, include_anomaly=False):
    """
    Generate a single synthetic orbital parameter sequence.
    
    Simulates realistic LEO orbit evolution with:
    - Atmospheric drag causing semi-major axis decay
    - J2 perturbations affecting RAAN and argument of perigee
    - Gaussian noise to simulate telemetry errors
    - Occasional larger perturbations (maneuvers, solar activity)
    
    Returns:
        sequence: (TOTAL_TIMESTEPS, 6) array of orbital elements
        risk_label: 0 (low risk) or 1 (high risk based on predicted decay)
    """
    # Initialize orbital elements with small random variations
    a = INITIAL_SEMI_MAJOR_AXIS + np.random.uniform(-20, 20)
    e = INITIAL_ECCENTRICITY + np.random.uniform(0, 0.005)
    i = INITIAL_INCLINATION + np.random.uniform(-0.05, 0.05)
    raan = INITIAL_RAAN + np.random.uniform(0, 2*np.pi)
    omega = INITIAL_ARG_PERIGEE + np.random.uniform(0, 2*np.pi)
    M = INITIAL_MEAN_ANOMALY + np.random.uniform(0, 2*np.pi)
    
    # Randomize decay rate for diversity
    # Higher decay rate = more aggressive orbital decay
    base_decay_rate = np.random.uniform(0.01, 0.08)  # km per timestep
    
    # For some sequences, create rapid decay scenarios
    if include_anomaly or np.random.random() < 0.2:
        base_decay_rate *= np.random.uniform(1.5, 3.0)
    
    sequence = np.zeros((TOTAL_TIMESTEPS, NUM_FEATURES))
    dt = 1.0  # Normalized timestep (could represent orbital periods or hours)
    
    for t in range(TOTAL_TIMESTEPS):
        # Store current state
        sequence[t] = [a, e, i, raan, omega, M]
        
        # Apply atmospheric drag decay to semi-major axis
        drag_decay = compute_atmospheric_drag_decay(a, e, t, base_decay_rate)
        a -= drag_decay
        
        # Ensure semi-major axis doesn't go below Earth's surface
        a = max(a, EARTH_RADIUS_KM + 150)
        
        # Apply J2 perturbations
        delta_raan, delta_omega = compute_j2_perturbations(a, e, i, dt)
        raan = (raan + delta_raan) % (2 * np.pi)
        omega = (omega + delta_omega) % (2 * np.pi)
        
        # Mean anomaly propagation (simplified: advances based on mean motion)
        n = np.sqrt(MU_EARTH / a**3)  # Mean motion
        M = (M + n * dt * 3600) % (2 * np.pi)  # Convert to appropriate units
        
        # Small random perturbations to eccentricity (drag effects)
        e += np.random.normal(0, 0.0001)
        e = np.clip(e, 0.0001, 0.05)  # Keep eccentricity in realistic range
        
        # Small inclination perturbations
        i += np.random.normal(0, 0.0001)
        i = np.clip(i, np.radians(40), np.radians(60))
        
        # Add Gaussian noise to simulate telemetry errors
        noise_scale = np.array([0.5, 0.0005, 0.001, 0.01, 0.01, 0.05])
        sequence[t] += np.random.normal(0, noise_scale)
        
        # Occasional larger perturbations (e.g., solar storms, small maneuvers)
        if np.random.random() < 0.02:
            sequence[t, 0] += np.random.normal(0, 2.0)  # a perturbation
            sequence[t, 1] += np.random.normal(0, 0.002)  # e perturbation
    
    # Determine collision risk label based on predicted semi-major axis
    # High risk if semi-major axis drops below threshold in prediction horizon
    min_predicted_a = np.min(sequence[INPUT_TIMESTEPS:, 0])
    risk_label = 1 if min_predicted_a < HIGH_RISK_A_THRESHOLD else 0
    
    return sequence, risk_label


def generate_dataset():
    """
    Generate the complete synthetic dataset of orbital sequences.
    
    Returns:
        sequences: (NUM_SEQUENCES, TOTAL_TIMESTEPS, NUM_FEATURES)
        risk_labels: (NUM_SEQUENCES,) binary labels
    """
    print("Generating synthetic orbital data...")
    sequences = np.zeros((NUM_SEQUENCES, TOTAL_TIMESTEPS, NUM_FEATURES))
    risk_labels = np.zeros(NUM_SEQUENCES)
    
    # Ensure balanced risk labels by forcing some high-risk scenarios
    high_risk_count = 0
    target_high_risk = int(NUM_SEQUENCES * 0.35)  # Target ~35% high risk
    
    for i in range(NUM_SEQUENCES):
        # Force anomaly to create more high-risk cases initially
        force_anomaly = (high_risk_count < target_high_risk) and (i < NUM_SEQUENCES * 0.5)
        
        sequence, risk = generate_single_sequence(i, include_anomaly=force_anomaly)
        sequences[i] = sequence
        risk_labels[i] = risk
        
        if risk == 1:
            high_risk_count += 1
        
        if (i + 1) % 500 == 0:
            print(f"  Generated {i + 1}/{NUM_SEQUENCES} sequences")
    
    print(f"Dataset generated: {NUM_SEQUENCES} sequences")
    print(f"High-risk sequences: {int(np.sum(risk_labels))} ({100*np.mean(risk_labels):.1f}%)")
    
    return sequences, risk_labels


def normalize_data(sequences):
    """
    Normalize orbital elements to similar scales for neural network training.
    
    Returns normalized sequences and normalization parameters for inverse transform.
    """
    # Compute mean and std for each feature across all sequences and timesteps
    mean = np.mean(sequences, axis=(0, 1))
    std = np.std(sequences, axis=(0, 1))
    
    # Avoid division by zero
    std[std < 1e-8] = 1.0
    
    normalized = (sequences - mean) / std
    
    return normalized, mean, std


def denormalize_data(normalized_sequences, mean, std):
    """Inverse normalization to get original scale values."""
    return normalized_sequences * std + mean


# ============================================================================
# SECTION 3: PYTORCH DATASET AND DATALOADER
# ============================================================================

class OrbitDataset(Dataset):
    """
    PyTorch Dataset for orbital time-series data.
    
    Returns:
        input_seq: (INPUT_TIMESTEPS, NUM_FEATURES) - past observations
        target_seq: (PREDICTION_HORIZON, NUM_FEATURES) - future to predict
        risk_label: binary collision risk label
    """
    
    def __init__(self, sequences, risk_labels):
        self.sequences = torch.FloatTensor(sequences)
        self.risk_labels = torch.FloatTensor(risk_labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        full_seq = self.sequences[idx]
        input_seq = full_seq[:INPUT_TIMESTEPS]
        target_seq = full_seq[INPUT_TIMESTEPS:]
        risk_label = self.risk_labels[idx]
        
        return input_seq, target_seq, risk_label


# ============================================================================
# SECTION 4: BI-GRU MODEL ARCHITECTURE
# ============================================================================

class BiGRUOrbitPredictor(nn.Module):
    """
    Bidirectional GRU model for multi-step orbital parameter forecasting.
    
    Architecture:
    - Input: (batch, INPUT_TIMESTEPS, NUM_FEATURES)
    - Bi-GRU: 2 layers, hidden_dim=128, bidirectional
    - Two output heads:
      1. Regression head: predicts (PREDICTION_HORIZON, NUM_FEATURES)
      2. Classification head: predicts collision risk probability
    """
    
    def __init__(self, input_dim, hidden_dim, num_layers, output_steps, output_dim):
        super(BiGRUOrbitPredictor, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_steps = output_steps
        self.output_dim = output_dim
        
        # Bidirectional GRU encoder
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        # Layer normalization for stable training
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        
        # Regression head: projects GRU output to prediction horizon
        # Takes final hidden state and produces multi-step forecast
        self.fc_regression = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, output_steps * output_dim)
        )
        
        # Classification head: predicts collision risk from sequence representation
        self.fc_classification = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Attention mechanism for weighted combination of time steps
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        """
        Forward pass through the Bi-GRU network.
        
        Args:
            x: (batch, INPUT_TIMESTEPS, NUM_FEATURES)
        
        Returns:
            predictions: (batch, PREDICTION_HORIZON, NUM_FEATURES)
            risk_prob: (batch,) collision risk probabilities
        """
        batch_size = x.size(0)
        
        # Pass through Bi-GRU
        # gru_out: (batch, seq_len, hidden_dim * 2) - bidirectional concatenation
        # hidden: (num_layers * 2, batch, hidden_dim) - final hidden states
        gru_out, hidden = self.gru(x)
        
        # Apply attention mechanism to get weighted sequence representation
        attention_weights = self.attention(gru_out)  # (batch, seq_len, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Weighted sum of hidden states
        context = torch.sum(attention_weights * gru_out, dim=1)  # (batch, hidden_dim * 2)
        context = self.layer_norm(context)
        
        # Regression: predict future orbital elements
        reg_output = self.fc_regression(context)  # (batch, output_steps * output_dim)
        predictions = reg_output.view(batch_size, self.output_steps, self.output_dim)
        
        # Classification: predict collision risk
        risk_prob = self.fc_classification(context).squeeze(-1)  # (batch,)
        
        return predictions, risk_prob


# ============================================================================
# SECTION 5: TRAINING AND EVALUATION FUNCTIONS
# ============================================================================

def train_epoch(model, dataloader, criterion_mse, criterion_bce, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0.0
    total_mse_loss = 0.0
    total_bce_loss = 0.0
    
    for batch_idx, (input_seq, target_seq, risk_labels) in enumerate(dataloader):
        input_seq = input_seq.to(device)
        target_seq = target_seq.to(device)
        risk_labels = risk_labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        predictions, risk_probs = model(input_seq)
        
        # Compute losses
        mse_loss = criterion_mse(predictions, target_seq)
        bce_loss = criterion_bce(risk_probs, risk_labels)
        
        # Combined loss (weighted)
        loss = mse_loss + 0.5 * bce_loss
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        total_mse_loss += mse_loss.item()
        total_bce_loss += bce_loss.item()
    
    n_batches = len(dataloader)
    return total_loss / n_batches, total_mse_loss / n_batches, total_bce_loss / n_batches


def evaluate(model, dataloader, criterion_mse, criterion_bce, device):
    """Evaluate the model on validation/test data."""
    model.eval()
    total_loss = 0.0
    total_mse_loss = 0.0
    total_bce_loss = 0.0
    
    all_predictions = []
    all_targets = []
    all_risk_probs = []
    all_risk_labels = []
    
    with torch.no_grad():
        for input_seq, target_seq, risk_labels in dataloader:
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            risk_labels = risk_labels.to(device)
            
            predictions, risk_probs = model(input_seq)
            
            mse_loss = criterion_mse(predictions, target_seq)
            bce_loss = criterion_bce(risk_probs, risk_labels)
            loss = mse_loss + 0.5 * bce_loss
            
            total_loss += loss.item()
            total_mse_loss += mse_loss.item()
            total_bce_loss += bce_loss.item()
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(target_seq.cpu().numpy())
            all_risk_probs.append(risk_probs.cpu().numpy())
            all_risk_labels.append(risk_labels.cpu().numpy())
    
    n_batches = len(dataloader)
    
    # Concatenate all batches
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_risk_probs = np.concatenate(all_risk_probs, axis=0)
    all_risk_labels = np.concatenate(all_risk_labels, axis=0)
    
    return (
        total_loss / n_batches,
        total_mse_loss / n_batches,
        total_bce_loss / n_batches,
        all_predictions,
        all_targets,
        all_risk_probs,
        all_risk_labels
    )


def compute_metrics(predictions, targets, risk_probs, risk_labels, mean, std):
    """
    Compute evaluation metrics.
    
    Returns MAE and RMSE for semi-major axis, classification accuracy, and AUC.
    """
    # Denormalize predictions and targets
    predictions_denorm = denormalize_data(predictions, mean, std)
    targets_denorm = denormalize_data(targets, mean, std)
    
    # Semi-major axis is feature index 0
    pred_a = predictions_denorm[:, :, 0]
    true_a = targets_denorm[:, :, 0]
    
    # MAE and RMSE for semi-major axis
    mae = np.mean(np.abs(pred_a - true_a))
    rmse = np.sqrt(np.mean((pred_a - true_a) ** 2))
    
    # Classification metrics
    risk_preds = (risk_probs >= 0.5).astype(int)
    accuracy = accuracy_score(risk_labels, risk_preds)
    
    # AUC-ROC
    try:
        auc = roc_auc_score(risk_labels, risk_probs)
    except ValueError:
        auc = 0.5  # If only one class present
    
    return mae, rmse, accuracy, auc, predictions_denorm, targets_denorm


# ============================================================================
# SECTION 6: VISUALIZATION FUNCTIONS
# ============================================================================

def plot_training_curves(train_losses, val_losses, train_mse, val_mse, train_bce, val_bce):
    """Plot training and validation loss curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Total loss
    axes[0].plot(epochs, train_losses, 'b-', label='Train', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Total Loss', fontsize=12)
    axes[0].set_title('Combined Loss (MSE + 0.5×BCE)', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # MSE loss
    axes[1].plot(epochs, train_mse, 'b-', label='Train', linewidth=2)
    axes[1].plot(epochs, val_mse, 'r-', label='Validation', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('MSE Loss', fontsize=12)
    axes[1].set_title('Orbital Parameter MSE Loss', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # BCE loss
    axes[2].plot(epochs, train_bce, 'b-', label='Train', linewidth=2)
    axes[2].plot(epochs, val_bce, 'r-', label='Validation', linewidth=2)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('BCE Loss', fontsize=12)
    axes[2].set_title('Collision Risk BCE Loss', fontsize=14)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: training_curves.png")


def plot_orbit_predictions(predictions, targets, input_sequences, mean, std, indices):
    """
    Plot true vs predicted semi-major axis evolution for selected sequences.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Denormalize
    predictions_denorm = denormalize_data(predictions, mean, std)
    targets_denorm = denormalize_data(targets, mean, std)
    input_denorm = denormalize_data(input_sequences, mean, std)
    
    for idx, (ax, seq_idx) in enumerate(zip(axes, indices)):
        # Full sequence: input + target
        time_input = np.arange(INPUT_TIMESTEPS)
        time_pred = np.arange(INPUT_TIMESTEPS, TOTAL_TIMESTEPS)
        
        # Semi-major axis (feature 0)
        input_a = input_denorm[seq_idx, :, 0]
        true_a = targets_denorm[seq_idx, :, 0]
        pred_a = predictions_denorm[seq_idx, :, 0]
        
        # Convert to altitude
        input_alt = input_a - EARTH_RADIUS_KM
        true_alt = true_a - EARTH_RADIUS_KM
        pred_alt = pred_a - EARTH_RADIUS_KM
        
        ax.plot(time_input, input_alt, 'b-', linewidth=2, label='Input (Observed)')
        ax.plot(time_pred, true_alt, 'g-', linewidth=2, label='True Future')
        ax.plot(time_pred, pred_alt, 'r--', linewidth=2, label='Predicted')
        
        # Mark danger zone
        ax.axhline(y=450, color='orange', linestyle=':', linewidth=1.5, label='Danger Zone (450 km)')
        ax.axhline(y=480, color='purple', linestyle=':', linewidth=1.5, label='Risk Threshold (480 km)')
        
        ax.set_xlabel('Timestep', fontsize=11)
        ax.set_ylabel('Altitude (km)', fontsize=11)
        ax.set_title(f'Sequence {seq_idx + 1}: Orbit Altitude Forecast', fontsize=12)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('orbit_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: orbit_predictions.png")


def plot_3d_orbits(predictions, targets, mean, std, normal_idx, decaying_idx):
    """
    3D visualization of predicted orbits for normal vs decaying satellite.
    
    Uses simplified Keplerian elements to compute 3D positions.
    """
    fig = plt.figure(figsize=(14, 6))
    
    # Denormalize
    predictions_denorm = denormalize_data(predictions, mean, std)
    
    for plot_idx, (seq_idx, title) in enumerate([(normal_idx, 'Stable Orbit'), 
                                                   (decaying_idx, 'Decaying Orbit')]):
        ax = fig.add_subplot(1, 2, plot_idx + 1, projection='3d')
        
        # Extract orbital elements
        a = predictions_denorm[seq_idx, :, 0]  # Semi-major axis
        e = np.clip(predictions_denorm[seq_idx, :, 1], 0.001, 0.1)  # Eccentricity
        i = predictions_denorm[seq_idx, :, 2]  # Inclination
        raan = predictions_denorm[seq_idx, :, 3]  # RAAN
        omega = predictions_denorm[seq_idx, :, 4]  # Arg of perigee
        M = predictions_denorm[seq_idx, :, 5]  # Mean anomaly
        
        # Compute 3D positions
        positions = []
        for t in range(len(a)):
            # Simplified true anomaly (for visualization)
            nu = M[t]  # Approximation for near-circular orbits
            
            # Radius at this point
            r = a[t] * (1 - e[t]**2) / (1 + e[t] * np.cos(nu))
            
            # Position in orbital plane
            x_orbital = r * np.cos(nu)
            y_orbital = r * np.sin(nu)
            
            # Rotation matrices for 3D position
            cos_raan, sin_raan = np.cos(raan[t]), np.sin(raan[t])
            cos_i, sin_i = np.cos(i[t]), np.sin(i[t])
            cos_omega, sin_omega = np.cos(omega[t]), np.sin(omega[t])
            
            # Transform to ECI coordinates (simplified)
            x = (cos_raan * cos_omega - sin_raan * sin_omega * cos_i) * x_orbital + \
                (-cos_raan * sin_omega - sin_raan * cos_omega * cos_i) * y_orbital
            y = (sin_raan * cos_omega + cos_raan * sin_omega * cos_i) * x_orbital + \
                (-sin_raan * sin_omega + cos_raan * cos_omega * cos_i) * y_orbital
            z = sin_omega * sin_i * x_orbital + cos_omega * sin_i * y_orbital
            
            positions.append([x, y, z])
        
        positions = np.array(positions)
        
        # Plot orbit trajectory with color gradient (time evolution)
        colors = plt.cm.plasma(np.linspace(0, 1, len(positions)))
        for j in range(len(positions) - 1):
            ax.plot(positions[j:j+2, 0], positions[j:j+2, 1], positions[j:j+2, 2],
                   color=colors[j], linewidth=2)
        
        # Plot Earth (scaled representation)
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        earth_scale = EARTH_RADIUS_KM
        x_earth = earth_scale * np.cos(u) * np.sin(v)
        y_earth = earth_scale * np.sin(u) * np.sin(v)
        z_earth = earth_scale * np.cos(v)
        ax.plot_surface(x_earth, y_earth, z_earth, color='blue', alpha=0.3)
        
        ax.set_xlabel('X (km)', fontsize=10)
        ax.set_ylabel('Y (km)', fontsize=10)
        ax.set_zlabel('Z (km)', fontsize=10)
        ax.set_title(f'{title} - 3D Trajectory', fontsize=12)
        
        # Add colorbar for time
        sm = plt.cm.ScalarMappable(cmap='plasma', 
                                    norm=plt.Normalize(vmin=0, vmax=PREDICTION_HORIZON))
        sm.set_array([])
    
    plt.tight_layout()
    plt.savefig('3d_orbits.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: 3d_orbits.png")


def plot_decay_and_risk_histograms(predictions, risk_probs, mean, std):
    """
    Plot histograms of predicted decay rates and collision risk scores.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Denormalize predictions
    predictions_denorm = denormalize_data(predictions, mean, std)
    
    # Compute decay rates (change in semi-major axis over prediction horizon)
    # Decay rate = (initial - final) / time
    initial_a = predictions_denorm[:, 0, 0]
    final_a = predictions_denorm[:, -1, 0]
    decay_rates = (initial_a - final_a) / PREDICTION_HORIZON  # km per timestep
    
    # Decay rate histogram
    axes[0].hist(decay_rates, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    axes[0].axvline(x=np.mean(decay_rates), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {np.mean(decay_rates):.3f} km/step')
    axes[0].set_xlabel('Decay Rate (km/timestep)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Distribution of Predicted Orbit Decay Rates', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Risk score histogram
    axes[1].hist(risk_probs, bins=50, color='coral', edgecolor='white', alpha=0.8)
    axes[1].axvline(x=0.5, color='red', linestyle='--', linewidth=2, 
                    label='Classification Threshold')
    axes[1].set_xlabel('Collision Risk Probability', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Distribution of Collision Risk Scores', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('decay_risk_histograms.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: decay_risk_histograms.png")


def plot_confusion_matrix(risk_labels, risk_probs):
    """
    Plot confusion matrix for collision risk classification.
    """
    risk_preds = (risk_probs >= 0.5).astype(int)
    
    cm = confusion_matrix(risk_labels, risk_preds)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                   display_labels=['Low Risk', 'High Risk'])
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    
    ax.set_title('Collision Risk Classification\nConfusion Matrix', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: confusion_matrix.png")


# ============================================================================
# SECTION 7: MAIN EXECUTION
# ============================================================================

def main():
    """Main function to run the complete pipeline."""
    
    print("=" * 70)
    print("SATELLITE ORBIT DRIFT PREDICTOR WITH BI-GRU")
    print("Portfolio Project for Dhruva Space")
    print("=" * 70)
    print()
    
    # ========================================
    # Step 1: Generate Synthetic Data
    # ========================================
    print("STEP 1: Generating Synthetic Orbital Data")
    print("-" * 50)
    
    sequences, risk_labels = generate_dataset()
    
    # Normalize data
    normalized_sequences, mean, std = normalize_data(sequences)
    
    print(f"Data shape: {sequences.shape}")
    print(f"Normalization - Mean: {mean[:3]}... Std: {std[:3]}...")
    print()
    
    # ========================================
    # Step 2: Create Train/Val/Test Splits
    # ========================================
    print("STEP 2: Creating Data Splits")
    print("-" * 50)
    
    # Shuffle data
    indices = np.random.permutation(NUM_SEQUENCES)
    shuffled_seq = normalized_sequences[indices]
    shuffled_labels = risk_labels[indices]
    
    # Split
    train_seq = shuffled_seq[:TRAIN_SIZE]
    train_labels = shuffled_labels[:TRAIN_SIZE]
    
    val_seq = shuffled_seq[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]
    val_labels = shuffled_labels[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]
    
    test_seq = shuffled_seq[TRAIN_SIZE + VAL_SIZE:]
    test_labels = shuffled_labels[TRAIN_SIZE + VAL_SIZE:]
    
    print(f"Train: {len(train_seq)} sequences ({100*np.mean(train_labels):.1f}% high-risk)")
    print(f"Val:   {len(val_seq)} sequences ({100*np.mean(val_labels):.1f}% high-risk)")
    print(f"Test:  {len(test_seq)} sequences ({100*np.mean(test_labels):.1f}% high-risk)")
    print()
    
    # Create datasets and dataloaders
    train_dataset = OrbitDataset(train_seq, train_labels)
    val_dataset = OrbitDataset(val_seq, val_labels)
    test_dataset = OrbitDataset(test_seq, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # ========================================
    # Step 3: Initialize Model
    # ========================================
    print("STEP 3: Initializing Bi-GRU Model")
    print("-" * 50)
    
    model = BiGRUOrbitPredictor(
        input_dim=NUM_FEATURES,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        output_steps=PREDICTION_HORIZON,
        output_dim=NUM_FEATURES
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model Architecture: Bi-GRU with {NUM_LAYERS} layers, hidden_dim={HIDDEN_DIM}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print()
    
    # ========================================
    # Step 4: Training
    # ========================================
    print("STEP 4: Training Model")
    print("-" * 50)
    
    criterion_mse = nn.MSELoss()
    criterion_bce = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                       factor=0.5, patience=5)
    
    train_losses, val_losses = [], []
    train_mse_losses, val_mse_losses = [], []
    train_bce_losses, val_bce_losses = [], []
    best_val_loss = float('inf')
    
    for epoch in range(1, NUM_EPOCHS + 1):
        # Train
        train_loss, train_mse, train_bce = train_epoch(
            model, train_loader, criterion_mse, criterion_bce, optimizer, device
        )
        
        # Validate
        val_loss, val_mse, val_bce, _, _, _, _ = evaluate(
            model, val_loader, criterion_mse, criterion_bce, device
        )
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Store losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_mse_losses.append(train_mse)
        val_mse_losses.append(val_mse)
        train_bce_losses.append(train_bce)
        val_bce_losses.append(val_bce)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
        
        # Progress logging
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{NUM_EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} (MSE: {train_mse:.4f}, BCE: {train_bce:.4f}) | "
                  f"Val Loss: {val_loss:.4f} (MSE: {val_mse:.4f}, BCE: {val_bce:.4f})")
    
    print()
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print()
    
    # ========================================
    # Step 5: Evaluation on Test Set
    # ========================================
    print("STEP 5: Evaluating on Test Set")
    print("-" * 50)
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Evaluate
    test_loss, test_mse, test_bce, predictions, targets, risk_probs, test_risk_labels = evaluate(
        model, test_loader, criterion_mse, criterion_bce, device
    )
    
    # Compute metrics
    mae, rmse, accuracy, auc, pred_denorm, target_denorm = compute_metrics(
        predictions, targets, risk_probs, test_risk_labels, mean, std
    )
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"  - MSE: {test_mse:.4f}")
    print(f"  - BCE: {test_bce:.4f}")
    print()
    print("Semi-Major Axis Forecast Metrics:")
    print(f"  - MAE:  {mae:.4f} km")
    print(f"  - RMSE: {rmse:.4f} km")
    print()
    print("Collision Risk Classification Metrics:")
    print(f"  - Accuracy: {accuracy:.4f} ({100*accuracy:.1f}%)")
    print(f"  - AUC-ROC:  {auc:.4f}")
    print()
    
    # ========================================
    # Step 6: Visualizations
    # ========================================
    print("STEP 6: Generating Visualizations")
    print("-" * 50)
    
    # Get input sequences for visualization
    test_input_sequences = test_seq[:, :INPUT_TIMESTEPS, :]
    
    # 1. Training curves
    print("Plotting training curves...")
    plot_training_curves(train_losses, val_losses, 
                         train_mse_losses, val_mse_losses,
                         train_bce_losses, val_bce_losses)
    
    # 2. Orbit predictions (3 examples)
    print("Plotting orbit predictions...")
    example_indices = [0, len(predictions)//2, len(predictions)-1]
    plot_orbit_predictions(predictions, targets, test_input_sequences, 
                          mean, std, example_indices)
    
    # 3. 3D orbit visualization
    print("Plotting 3D orbits...")
    # Find indices for normal (high predicted altitude) and decaying (low altitude) orbits
    final_altitudes = pred_denorm[:, -1, 0] - EARTH_RADIUS_KM
    normal_idx = np.argmax(final_altitudes)  # Highest final altitude
    decaying_idx = np.argmin(final_altitudes)  # Lowest final altitude
    plot_3d_orbits(predictions, targets, mean, std, normal_idx, decaying_idx)
    
    # 4. Decay and risk histograms
    print("Plotting decay and risk histograms...")
    plot_decay_and_risk_histograms(predictions, risk_probs, mean, std)
    
    # 5. Confusion matrix
    print("Plotting confusion matrix...")
    plot_confusion_matrix(test_risk_labels, risk_probs)
    
    print()
    print("=" * 70)
    print("TRAINING AND EVALUATION COMPLETE")
    print("=" * 70)
    print()
    print("Generated files:")
    print("  - best_model.pth (trained model weights)")
    print("  - training_curves.png")
    print("  - orbit_predictions.png")
    print("  - 3d_orbits.png")
    print("  - decay_risk_histograms.png")
    print("  - confusion_matrix.png")
    print()
    
    # Summary statistics
    print("SUMMARY STATISTICS")
    print("-" * 50)
    print(f"Dataset: {NUM_SEQUENCES} synthetic LEO orbit sequences")
    print(f"Input: {INPUT_TIMESTEPS} timesteps × {NUM_FEATURES} orbital elements")
    print(f"Output: {PREDICTION_HORIZON} timesteps forecast + risk classification")
    print(f"Model: Bi-GRU with {NUM_LAYERS} layers, {HIDDEN_DIM} hidden units")
    print(f"Best Semi-Major Axis MAE: {mae:.2f} km")
    print(f"Collision Risk Accuracy: {100*accuracy:.1f}%")
    print(f"Collision Risk AUC: {auc:.3f}")


if __name__ == "__main__":
    main()
