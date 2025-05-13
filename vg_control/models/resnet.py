import os
import tqdm
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Literal
from torch.utils.data import DataLoader
from datetime import datetime


# --------------- Training

# --- Data parameters
DATASET_SIZE_TRAIN, DATASET_SIZE_VAL = 5000, 1000
BATCH_SIZE = 128
VIDEO_DURATION = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model parameters
RESNET_TYPE = 'resnet18'
PRETRAINED = True
DROPOUT_P = 0.2
NUM_DOTS: Literal[1, 3] = 1

# --- Optimization parameters
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
MOMENTUM = 0.9

# --- Training parameters
NUM_EPOCHS = 5
VAL_EVERY = 1


class RGB_ResNetInverseDynamics(nn.Module):
    def __init__(self,
                 kernel_size=7,
                 padding=3,
                 output_dim=2 * NUM_DOTS,
                 pretrained=True,
                 resnet_type='resnet18',
                 dropout_p=0.2):
        super().__init__()
        self.pretrained = pretrained

        if resnet_type == 'resnet18':
            self.resnet = models.resnet18(pretrained=pretrained)
            feature_dim = 512
        elif resnet_type == 'resnet34':
            self.resnet = models.resnet34(pretrained=pretrained)
            feature_dim = 512
        elif resnet_type == 'resnet50':
            self.resnet = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        elif resnet_type == 'resnet101':
            self.resnet = models.resnet101(pretrained=pretrained)
            feature_dim = 2048
        else:
            raise ValueError(f'Invalid resnet type: {resnet_type}')

        # Adapt the first conv layer for RGB input
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=kernel_size, stride=2, padding=padding, bias=False)
        # Remove classification head
        self.resnet.fc = nn.Identity()
        
        # Feature extractor (shared for both frames)
        self.feature_extractor = nn.Sequential(
            self.resnet,
            nn.Flatten()
        )

        # Regressor head with dropout
        # Takes features from both frames (concatenated)
        self.regressor = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),  # Doubled due to concatenation
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, frame1, frame2):
        """
        Forward pass for inverse dynamics:
        - Extract features from both frames
        - Concatenate features
        - Predict the action that caused the transition
        
        If predict_per_dot is True and num_dots is provided, predicts
        separate actions for each dot (useful for multi-dot datasets)
        
        Args:
            frame1: First frame [batch, channels, height, width]
            frame2: Second frame [batch, channels, height, width]
            num_dots: Number of dots to predict actions for (optional)
            
        Returns:
            actions: Predicted actions with shape:
                    - [batch, num_dots, 2]
        """
        # Extract features from both frames
        B, *_ = frame1.shape
        features1 = self.feature_extractor(frame1)
        features2 = self.feature_extractor(frame2)
        
        # Concatenate features
        combined_features = torch.cat((features1, features2), dim=1)
        
        # Predict actions
        actions = self.regressor(combined_features)

        actions = actions.view(B, NUM_DOTS, 2)
        
        return actions

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


def make_dataloaders(dataset_type: Literal['single', 'multiple']):
    from vg_control.data.movingdot.single import ContinuousMotionDataset
    from vg_control.data.movingdot.multiple import create_three_datasets

    train_ds, val_ds = None, None
    if dataset_type == 'single':
        train_ds = ContinuousMotionDataset(size=DATASET_SIZE_TRAIN, batch_size=BATCH_SIZE, n_steps=VIDEO_DURATION, train=True, device=DEVICE)
        val_ds = ContinuousMotionDataset(size=DATASET_SIZE_VAL, batch_size=BATCH_SIZE, n_steps=VIDEO_DURATION, train=False, device=DEVICE)
    elif dataset_type == 'multiple':
        train_ds = create_three_datasets(size=DATASET_SIZE_TRAIN, batch_size=BATCH_SIZE, n_steps=VIDEO_DURATION, train=True, device=DEVICE)
        val_ds = create_three_datasets(size=DATASET_SIZE_VAL, batch_size=BATCH_SIZE, n_steps=VIDEO_DURATION, train=False, device=DEVICE)
    else:
        raise ValueError(f'Invalid dataset type: {dataset_type}')

    # dataset handles batching :)
    train_dl = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False)
    return train_dl, val_dl

def make_model(resnet_type: Literal['resnet18', 'resnet34', 'resnet50', 'resnet101'], dropout_p: float = 0.2):
    model = RGB_ResNetInverseDynamics(resnet_type=resnet_type, dropout_p=dropout_p)
    model.to(DEVICE)
    return model

def make_optimizer(model: nn.Module, learning_rate: float = 0.001, weight_decay: float = 0.0001, momentum: float = 0.9):
    return torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)

def make_scheduler(optimizer: torch.optim.Optimizer, num_epochs: int, train_dl: DataLoader):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_dl))

def save_checkpoint(model, optimizer, epoch, loss, path):
    """
    Save model checkpoint
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")

def load_checkpoint(model, optimizer, path):
    """
    Load model checkpoint
    """
    if not os.path.exists(path):
        print(f"No checkpoint found at {path}")
        return 0, float('inf')
    
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"Loaded checkpoint from epoch {epoch} with loss {loss:.4f}")
    return epoch, loss

def process_batch_for_inverse_dynamics(batch, device):
    """
    Process a batch for inverse dynamics prediction.
    Separates consecutive frames and prepares them for model input.
    
    For inverse dynamics:
    - Input: Frame at t and Frame at t+1
    - Output: Action that caused transition from t to t+1
    
    The movingdot dataset returns Sample namedtuples with:
    - states: shape [dl_batch_size=1, ds_batch_size, time_steps, num_dots, height, width]
    - locations: shape [dl_batch_size=1, ds_batch_size, time_steps, num_dots, 2]
    - actions: shape [dl_batch_size=1, ds_batch_size, time_steps-1, num_dots, 2]
    
    Returns:
        frame1: First frame in sequence (state at time t)
        frame2: Second frame in sequence (state at time t+1)
        target_actions: Ground truth actions between frames
        num_dots: Number of dots in the scene (for multi-dot prediction)
    """
    # For the movingdot dataset, the batch is a Sample namedtuple
    # We need to extract states and actions
    states = batch.states.to(device).squeeze(0) # remove batch dimension from dataloader
    actions = batch.actions.to(device).squeeze(0) # remove batch dimension from dataloader
    
    num_dots = states.shape[-3]
    assert num_dots == NUM_DOTS, f"Number of dots in dataset ({num_dots}) does not match expected number of dots ({NUM_DOTS})"
    
    # For the inverse dynamics model, we need:
    # - frame at t (first frame in sequence)
    # - frame at t+1 (second frame in sequence) 
    # - actions that caused transition from t to t+1
    
    # Extract frames, ensuring RGB format
    # States shape: [batch, time, channels/dots, height, width]
    frame1 = states[:, 0]  # First frame (t)
    frame2 = states[:, 1]  # Second frame (t+1)
    actions = actions.squeeze(1) # assume we have 2 frames and 1 action b/w them always
    if num_dots == 3:
        pass
    if num_dots == 1:
        frame1 = frame1.repeat(1, 3, 1, 1)
        frame2 = frame2.repeat(1, 3, 1, 1)

    
    return frame1, frame2, actions

def train_step(model, optimizer, batch, criterion, device):
    """
    Execute one training step
    """
    model.train()
    optimizer.zero_grad()
    
    # Process batch for inverse dynamics
    frame1, frame2, target_actions = process_batch_for_inverse_dynamics(batch, device)
    
    # Forward pass - using BOTH frames to predict the action that caused the transition
    pred_actions = model(frame1, frame2)
    
    # Compute loss
    loss = criterion(pred_actions, target_actions)
    
    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    
    return loss.item()

def validate_step(model, batch, criterion, device):
    """
    Execute one validation step
    """
    model.eval()
    
    # Process batch for inverse dynamics
    frame1, frame2, target_actions = process_batch_for_inverse_dynamics(batch, device)
    
    with torch.no_grad():
        # Forward pass - using BOTH frames
        pred_actions = model(frame1, frame2)
        
        # Compute loss
        loss = criterion(pred_actions, target_actions)
        
        # Compute mean squared error
        mse = torch.mean((pred_actions - target_actions) ** 2)
    
    return loss.item(), mse.item()

def validate_epoch(model, val_loader, criterion, device):
    """
    Validate model for one epoch
    """
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    
    with torch.no_grad():
        val_pbar = tqdm.tqdm(val_loader, desc="Validating")
        for batch_idx, batch in enumerate(val_pbar):
            loss, mse = validate_step(model, batch, criterion, device)
            total_loss += loss
            total_mse += mse
            
            val_pbar.set_postfix({
                'val_loss': total_loss / (batch_idx + 1),
                'val_mse': total_mse / (batch_idx + 1)
            })
    
    avg_loss = total_loss / len(val_loader)
    avg_mse = total_mse / len(val_loader)
    
    print(f"Validation - Avg Loss: {avg_loss:.4f}, Avg MSE: {avg_mse:.4f}")
    
    return avg_loss, avg_mse

def train_epoch(model, optimizer, scheduler, train_loader, criterion, device):
    """
    Train model for one epoch
    """
    model.train()
    total_loss = 0.0
    
    train_pbar = tqdm.tqdm(train_loader, desc="Training")
    for batch_idx, batch in enumerate(train_pbar):
        loss = train_step(model, optimizer, batch, criterion, device)
        total_loss += loss
        
        # Update learning rate
        scheduler.step()
        
        # Update progress bar
        train_pbar.set_postfix({
            'train_loss': total_loss / (batch_idx + 1),
            'lr': scheduler.get_last_lr()[0]
        })
    
    avg_loss = total_loss / len(train_loader)
    print(f"Training - Avg Loss: {avg_loss:.4f}")
    
    return avg_loss

def train(model, optimizer, scheduler, train_dl, val_dl, num_epochs=NUM_EPOCHS, checkpoint_dir="checkpoints"):
    """
    Train the model
    """
    print(f"Training on device: {DEVICE}")
    print(f"Total training samples: {len(train_dl.dataset)}")
    print(f"Total validation samples: {len(val_dl.dataset)}")
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Define loss function for regression
    criterion = nn.MSELoss()
    
    # Best validation loss for model checkpointing
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train for one epoch
        train_loss = train_epoch(model, optimizer, scheduler, train_dl, criterion, DEVICE)
        
        # Validate if needed
        if epoch % VAL_EVERY == 0:
            val_loss, val_mse = validate_epoch(model, val_dl, criterion, DEVICE)
            
            # Save checkpoint if this is the best model so far
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                checkpoint_path = os.path.join(checkpoint_dir, f"best_model_{timestamp}.pth")
                save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)
        
        # Save regular checkpoint every epoch
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}_{timestamp}.pth")
        save_checkpoint(model, optimizer, epoch, train_loss, checkpoint_path)
    
    print(f"Training completed! Best validation loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    train_dl, val_dl = make_dataloaders(dataset_type='single')
    model = make_model(resnet_type=RESNET_TYPE, dropout_p=DROPOUT_P)
    optimizer = make_optimizer(model, learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM)
    scheduler = make_scheduler(optimizer, num_epochs=NUM_EPOCHS, train_dl=train_dl)
    
    train(model, optimizer, scheduler, train_dl, val_dl)
    