import torch
import numpy as np
import math
from module.Attention import *
from module.CPUEmbedding import *
from module.Common import *


# Dispose Loggers.
emphasis_logger = logging.getLogger('Emphasis')
emphasis_logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler(sys.stderr)
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))

file_handler = logging.FileHandler(os.path.join(LOG_ROOT, 'Emphasis.log'))
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))

emphasis_logger.addHandler(console_handler)
emphasis_logger.addHandler(file_handler)
emphasis_logger.info(
    'Construct logger for Emphasis succeeded, current working directory: %s, logs will be written in %s' %
    (os.getcwd(), LOG_ROOT))

# ============================================================================
# 1. TRANSFORMER-BASED MODEL (Most Powerful)
# ============================================================================

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x


class TransformerEmphasisPredictor(nn.Module):
    """
    Transformer-based model for emphasis prediction.
    Input: (batch, n, 300) where n can vary
    Output: (batch, n) emphasis probabilities
    """
    
    def __init__(self, 
                 vector_dim: int = 300,
                 n_fixed: int = 120,
                 hidden_dim: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        
        self.n_fixed = n_fixed
        self.vector_dim = vector_dim
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_projection = nn.Linear(vector_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Attention pooling for global context
        self.global_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Output heads
        self.local_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.context_head = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Combine heads
        self.combine = nn.Linear(2, 1)
        
        emphasis_logger.info("Emphasis model initialized.")
        emphasis_logger.info(f"  Vector Dim: {vector_dim}")
        emphasis_logger.info(f"  Fixed Length: {n_fixed}")
        emphasis_logger.info(f"  Hidden Dim: {hidden_dim}")
        emphasis_logger.info(f"  Number of Heads: {num_heads}")
        emphasis_logger.info(f"  Number of Layers: {num_layers}")
        emphasis_logger.info(f"  Dropout: {dropout}")
        emphasis_logger.info("====================================")
        
    def preprocess_input(self, x):
        """
        Ensure input has exactly n_fixed vectors.
        If longer: take first n_fixed
        If shorter: pad with zeros
        """
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        if seq_len > self.n_fixed:
            # Truncate to first n_fixed
            return x[:, :self.n_fixed, :]
        elif seq_len < self.n_fixed:
            # Pad with zeros
            pad_size = self.n_fixed - seq_len
            padding = torch.zeros(batch_size, pad_size, self.vector_dim, 
                                device=x.device, dtype=x.dtype)
            return torch.cat([x, padding], dim=1)
        else:
            return x
    
    def forward(self, x, return_attention=False):
        """
        Forward pass.
        
        Args:
            x: (batch, seq_len, 300) input tensor
            return_attention: If True, return attention weights
        
        Returns:
            probs: (batch, n_fixed) emphasis probabilities
            attention_weights: Optional attention weights
        """
        # Preprocess to fixed length
        x = self.preprocess_input(x)
        batch_size = x.size(0)
        
        # Project input
        x = self.input_projection(x)  # (batch, n_fixed, hidden_dim)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        # We'll capture attention weights if needed
        if return_attention:
            # Use custom forward to get attention
            attn_weights = []
            for layer in self.transformer_encoder.layers:
                x, attn = layer.self_attn(x, x, x, need_weights=True)
                attn_weights.append(attn)
                x = layer(x)
        else:
            x = self.transformer_encoder(x)
        
        # Get global context using attention pooling
        global_query = self.global_query.expand(batch_size, -1, -1)
        global_weights = torch.softmax(
            torch.bmm(global_query, x.transpose(1, 2)) / math.sqrt(self.hidden_dim),
            dim=-1
        )
        global_context = torch.bmm(global_weights, x)  # (batch, 1, hidden_dim)
        global_context = global_context.expand(-1, self.n_fixed, -1)
        
        # Local features
        local_features = x
        
        # Process with local head
        local_probs = self.local_head(local_features).squeeze(-1)  # (batch, n_fixed)
        
        # Process with context head (convolutional)
        x_conv = x.transpose(1, 2)  # (batch, hidden_dim, n_fixed)
        context_probs = self.context_head(x_conv).squeeze(1)  # (batch, n_fixed)
        
        # Combine predictions
        combined = torch.stack([local_probs, context_probs], dim=-1)  # (batch, n_fixed, 2)
        probs = self.combine(combined).squeeze(-1)  # (batch, n_fixed)
        
        if return_attention:
            return probs, attn_weights
        return probs
    
    def predict_emphasis(self, x, threshold=0.5):
        """
        Predict binary emphasis mask.
        
        Args:
            x: Input tensor
            threshold: Probability threshold
        
        Returns:
            emphasized_vectors: (batch, n_fixed, 300)
            binary_mask: (batch, n_fixed)
            probs: (batch, n_fixed)
        """
        with torch.no_grad():
            probs = self.forward(x)
            binary_mask = (probs > threshold).float()
            
            # Apply emphasis
            x_processed = self.preprocess_input(x)
            mask_expanded = binary_mask.unsqueeze(-1)
            # Note: k is not part of this model, you'll need to apply it externally
            emphasized = x_processed * (1 + mask_expanded)  # ×2 for emphasized
            
            return emphasized, binary_mask, probs


# ============================================================================
# 2. AUTOENCODER-BASED MODEL (Simpler)
# ============================================================================

class AutoencoderEmphasisPredictor(nn.Module):
    """
    Autoencoder-based model for emphasis prediction.
    Fixed n=120 vectors.
    """
    
    def __init__(self, 
                 n_fixed: int = 120,
                 vector_dim: int = 300,
                 latent_dim: int = 128,
                 k: float = 2.0):
        super().__init__()
        
        self.n_fixed = n_fixed
        self.vector_dim = vector_dim
        self.latent_dim = latent_dim
        self.k = k
        
        # ========== ENCODER ==========
        self.encoder = nn.Sequential(
            # Conv layers
            nn.Conv1d(vector_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # After pooling: 120 -> 60 -> 30
            nn.Flatten(),
            nn.Linear(30 * 128, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, latent_dim),
            nn.ReLU()
        )
        
        # ========== DECODER ==========
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 30 * 128),
            nn.ReLU(),
            nn.Unflatten(1, (128, 30)),  # (batch, 128, 30)
            nn.Upsample(scale_factor=2, mode='nearest'),  # 30 -> 60
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 60 -> 120
            nn.Conv1d(64, vector_dim, kernel_size=3, padding=1)
        )
        
        # ========== MASK PREDICTOR ==========
        self.mask_predictor = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_fixed),
            nn.Sigmoid()
        )
        
        # ========== RECONSTRUCTION HEAD ==========
        self.reconstruction_head = nn.Conv1d(vector_dim, vector_dim, kernel_size=1)
        
        emphasis_logger.info("AutoencoderEmphasisPredictor initialized.")
        emphasis_logger.info(f"  Vector Dim: {vector_dim}")
        emphasis_logger.info(f"  Fixed Length: {n_fixed}")
        emphasis_logger.info(f"  Latent Dim: {latent_dim}")
        emphasis_logger.info(f"  Emphasis Factor k: {k}")
        emphasis_logger.info("====================================")
        
    def preprocess_input(self, x):
        """Ensure input has exactly n_fixed vectors."""
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        if seq_len > self.n_fixed:
            # Truncate
            return x[:, :self.n_fixed, :]
        elif seq_len < self.n_fixed:
            # Pad
            pad_size = self.n_fixed - seq_len
            padding = torch.zeros(batch_size, pad_size, self.vector_dim,
                                device=x.device, dtype=x.dtype)
            return torch.cat([x, padding], dim=1)
        else:
            return x
    
    def encode(self, x):
        """Encode input to latent representation."""
        # Input shape: (batch, seq_len, vector_dim)
        # Conv1d expects (batch, channels, seq_len)
        x_processed = self.preprocess_input(x)
        x_conv = x_processed.transpose(1, 2)  # (batch, vector_dim, n_fixed)
        
        latent = self.encoder(x_conv)
        return latent
    
    def decode(self, latent):
        """Decode latent representation."""
        # Decoder returns (batch, vector_dim, n_fixed)
        decoded = self.decoder(latent)
        # Transpose back to (batch, n_fixed, vector_dim)
        decoded = decoded.transpose(1, 2)
        return decoded
    
    def forward(self, x):
        """
        Forward pass.
        
        Returns:
            reconstructed: Reconstructed vectors
            mask_probs: Emphasis mask probabilities
        """
        # Encode
        latent = self.encode(x)
        
        # Decode for reconstruction
        decoded = self.decode(latent)
        reconstructed = self.reconstruction_head(decoded.transpose(1, 2)).transpose(1, 2)
        
        # Predict mask
        mask_probs = self.mask_predictor(latent)
        
        return reconstructed, mask_probs
    
    def predict_emphasis(self, x, threshold=0.5, k=None):
        """
        Predict emphasis with optional emphasis factor k.
        """
        if k is None:
            k = self.k
        
        with torch.no_grad():
            # Get mask predictions
            _, mask_probs = self.forward(x)
            binary_mask = (mask_probs > threshold).float()
            
            # Apply emphasis
            x_processed = self.preprocess_input(x)
            mask_expanded = binary_mask.unsqueeze(-1)
            scaling = 1.0 + (k - 1.0) * mask_expanded
            emphasized = x_processed * scaling
            
            return emphasized, binary_mask, mask_probs


# ============================================================================
# 3. LSTM-BASED MODEL (Sequential)
# ============================================================================

class LSTMBidirectionalPredictor(nn.Module):
    """
    Bidirectional LSTM model for sequential emphasis prediction.
    """
    
    def __init__(self,
                 n_fixed: int = 120,
                 vector_dim: int = 300,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        
        self.n_fixed = n_fixed
        self.vector_dim = vector_dim
        self.hidden_dim = hidden_dim
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=vector_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,  # Bidirectional
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Output layers
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),  # LSTM + attention outputs
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        emphasis_logger.info("LSTMBidirectionalPredictor initialized.")
        emphasis_logger.info(f"  Vector Dim: {vector_dim}")
        emphasis_logger.info(f"  Fixed Length: {n_fixed}")
        emphasis_logger.info(f"  Hidden Dim: {hidden_dim}")
        emphasis_logger.info(f"  Number of Layers: {num_layers}")
        emphasis_logger.info(f"  Dropout: {dropout}")
        emphasis_logger.info("====================================")
        
    def preprocess_input(self, x):
        """Ensure fixed length of n_fixed."""
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        if seq_len > self.n_fixed:
            return x[:, :self.n_fixed, :]
        elif seq_len < self.n_fixed:
            pad_size = self.n_fixed - seq_len
            padding = torch.zeros(batch_size, pad_size, self.vector_dim,
                                device=x.device, dtype=x.dtype)
            return torch.cat([x, padding], dim=1)
        else:
            return x
    
    def forward(self, x, return_attention=False):
        """Forward pass."""
        x = self.preprocess_input(x)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # (batch, n_fixed, hidden_dim * 2)
        
        # Self-attention
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Combine LSTM and attention outputs
        combined = torch.cat([lstm_out, attn_out], dim=-1)
        
        # Apply output layer to each position
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Process each position
        probs = self.output_layer(combined)  # (batch, n_fixed, 1)
        probs = probs.squeeze(-1)  # (batch, n_fixed)
        
        if return_attention:
            return probs, attn_weights
        return probs
    
    def predict_emphasis(self, x, threshold=0.5, k=2.0):
        """Predict emphasis mask."""
        with torch.no_grad():
            probs = self.forward(x)
            binary_mask = (probs > threshold).float()
            
            # Apply emphasis
            x_processed = self.preprocess_input(x)
            mask_expanded = binary_mask.unsqueeze(-1)
            scaling = 1.0 + (k - 1.0) * mask_expanded
            emphasized = x_processed * scaling
            
            return emphasized, binary_mask, probs


# ============================================================================
# 4. CONVOLUTIONAL MODEL (Lightweight)
# ============================================================================

class ConvEmphasisPredictor(nn.Module):
    """
    Convolutional model for emphasis prediction.
    Fast and lightweight.
    """
    
    def __init__(self,
                 n_fixed: int = 120,
                 vector_dim: int = 300,
                 hidden_dim: int = 128):
        super().__init__()
        
        self.n_fixed = n_fixed
        self.vector_dim = vector_dim
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # First block
            nn.Conv1d(vector_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Second block
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Third block
            nn.Conv1d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.ReLU(),
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Decoder/upsampling
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='nearest'),
            nn.Conv1d(hidden_dim * 4, hidden_dim * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(hidden_dim, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        emphasis_logger.info("ConvEmphasisPredictor initialized.")
        emphasis_logger.info(f"  Vector Dim: {vector_dim}")
        emphasis_logger.info(f"  Fixed Length: {n_fixed}")
        emphasis_logger.info(f"  Hidden Dim: {hidden_dim}")
        emphasis_logger.info("====================================")
        
    def preprocess_input(self, x):
        """Ensure fixed length."""
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        if seq_len > self.n_fixed:
            return x[:, :self.n_fixed, :]
        elif seq_len < self.n_fixed:
            pad_size = self.n_fixed - seq_len
            padding = torch.zeros(batch_size, pad_size, self.vector_dim,
                                device=x.device, dtype=x.dtype)
            return torch.cat([x, padding], dim=1)
        else:
            return x
    
    def forward(self, x):
        """Forward pass."""
        x = self.preprocess_input(x)
        
        # Transpose for conv1d: (batch, channels, seq_len)
        x = x.transpose(1, 2)
        
        # Apply convolutional layers
        features = self.conv_layers(x)
        
        # Get global context
        global_context = self.global_pool(features)
        global_context = global_context.expand_as(features)
        
        # Combine local and global features
        combined = torch.cat([features, global_context], dim=1)
        
        # Decode to predictions
        probs = self.decoder(combined)
        probs = probs.squeeze(1)  # (batch, n_fixed)
        
        return probs
    
    def predict_emphasis(self, x, threshold=0.5, k=2.0):
        """Predict emphasis."""
        with torch.no_grad():
            probs = self.forward(x)
            binary_mask = (probs > threshold).float()
            
            # Apply emphasis
            x_processed = self.preprocess_input(x)
            mask_expanded = binary_mask.unsqueeze(-1)
            scaling = 1.0 + (k - 1.0) * mask_expanded
            emphasized = x_processed * scaling
            
            return emphasized, binary_mask, probs


# ============================================================================
# 5. TRAINING UTILITIES AND WRAPPER CLASS
# ============================================================================

class EmphasisModelTrainer:
    """Training utilities for emphasis prediction models."""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.BCELoss()
        
        emphasis_logger.info("EmphasisModelTrainer initialized.")
        emphasis_logger.info(f"  Device: {self.device}")
        emphasis_logger.info("====================================")
        
    def train_epoch(self, dataloader, optimizer, epoch, print_every=50):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        emphasis_logger.info("Starting training epoch.")
        emphasis_logger.info(f"  Epoch: {epoch}")
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # Move to device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            
            if isinstance(self.model, AutoencoderEmphasisPredictor):
                # Autoencoder has two outputs
                reconstructed, mask_probs = self.model(inputs)
                
                # Calculate reconstruction loss
                reconstruction_loss = F.mse_loss(reconstructed, 
                                                self.model.preprocess_input(inputs))
                
                # Calculate mask loss
                mask_loss = self.criterion(mask_probs, targets)
                
                # Combine losses
                loss = reconstruction_loss + mask_loss
            else:
                # Other models have single output
                probs = self.model(inputs)
                loss = self.criterion(probs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            with torch.no_grad():
                if isinstance(self.model, AutoencoderEmphasisPredictor):
                    preds = (mask_probs > 0.5).float()
                else:
                    preds = (probs > 0.5).float()
                
                correct = (preds == targets).float().sum()
                total_correct += correct.item()
                total_samples += targets.numel()
            
            total_loss += loss.item()
            
            # Print progress
            if (batch_idx + 1) % print_every == 0:
                accuracy = total_correct / total_samples
                print(f'Epoch {epoch}, Batch {batch_idx+1}/{len(dataloader)}, '
                      f'Loss: {loss.item():.4f}, Acc: {accuracy:.4f}')
        
        avg_loss = total_loss / len(dataloader)
        avg_accuracy = total_correct / total_samples
        
        return avg_loss, avg_accuracy
    
    def evaluate(self, dataloader):
        """Evaluate model."""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        emphasis_logger.info("Starting evaluation.")
        emphasis_logger.info(f"  Device: {self.device}")
        emphasis_logger.info("====================================")
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                if isinstance(self.model, AutoencoderEmphasisPredictor):
                    reconstructed, mask_probs = self.model(inputs)
                    reconstruction_loss = F.mse_loss(reconstructed, 
                                                    self.model.preprocess_input(inputs))
                    mask_loss = self.criterion(mask_probs, targets)
                    loss = reconstruction_loss + mask_loss
                    preds = (mask_probs > 0.5).float()
                else:
                    probs = self.model(inputs)
                    loss = self.criterion(probs, targets)
                    preds = (probs > 0.5).float()
                
                total_loss += loss.item()
                total_correct += (preds == targets).float().sum().item()
                total_samples += targets.numel()
        
        avg_loss = total_loss / len(dataloader)
        avg_accuracy = total_correct / total_samples
        
        return avg_loss, avg_accuracy
    
    def train(self, train_loader, val_loader, epochs=50, lr=0.001):
        """Full training loop."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, epoch + 1)
            
            # Validate
            val_loss, val_acc = self.evaluate(val_loader)
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
                torch.save(best_model_state, 'best_model.pth')
                print(f'  Saved best model with val loss: {val_loss:.4f}')
        
        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        return self.model


# ============================================================================
# 6. DATASET AND DATA LOADING
# ============================================================================

class EmphasisDataset(torch.utils.data.Dataset):
    """Dataset for emphasis prediction."""
    
    def __init__(self, sequences_list, masks_list, id2embed, n_fixed=120):
        """
        Args:
            vectors_list: List of numpy arrays of shape (n_i, 300)
            masks_list: List of numpy arrays of shape (n_i,)
            n_fixed: Fixed length to pad/truncate to
        """
        self.n_fixed = n_fixed
        self.vector_dim = 300
        
        # Build vectors_list from sequences_list using id2embed
        if len(sequences_list) != len(masks_list):
            raise ValueError("sequences_list and masks_list must have the same length")

        vectors_list = []

        # If id2embed is None assume sequences_list already contains vector arrays
        if id2embed is None:
            raise ValueError("id2embed cannot be None when sequences_list contains token IDs")
        else:
            for seq in sequences_list:
                temp = []
                for idx in seq:
                    if idx in id2embed.keys():
                        temp.append(id2embed[idx])
                    else:
                        temp.append(np.zeros((self.vector_dim,), dtype=np.float32))
                vectors_list.append(np.asarray(temp, dtype=np.float32))
                
        # Ensure masks_list entries are numpy float32 arrays
        masks_list = [np.asarray(m, dtype=np.float32) for m in masks_list]
        
        # Preprocess all data
        self.vectors = []
        self.masks = []
        
        for vectors, mask in zip(vectors_list, masks_list):
            # Process vectors
            if len(vectors) > n_fixed:
                vectors_processed = vectors[:n_fixed]
                mask_processed = mask[:n_fixed]
            elif len(vectors) < n_fixed:
                pad_len = n_fixed - len(vectors)
                vectors_processed = np.pad(vectors, ((0, pad_len), (0, 0)), mode='constant')
                mask_processed = np.pad(mask, (0, pad_len), mode='constant')
            else:
                vectors_processed = vectors
                mask_processed = mask
            
            self.vectors.append(vectors_processed)
            self.masks.append(mask_processed)
    
    def __len__(self):
        return len(self.vectors)
    
    def __getitem__(self, idx):
        vectors = torch.FloatTensor(self.vectors[idx])
        mask = torch.FloatTensor(self.masks[idx])
        return vectors, mask
    
    @staticmethod
    def create_synthetic_data(n_samples=1000, min_len=50, max_len=200, n_fixed=120):
        """Create synthetic dataset for testing."""
        vectors_list = []
        masks_list = []
        
        for _ in range(n_samples):
            # Random sequence length
            n_vectors = np.random.randint(min_len, max_len + 1)
            
            # Random vectors
            vectors = np.random.randn(n_vectors, 300).astype(np.float32)
            
            # Random mask (10-20% emphasized)
            mask = (np.random.rand(n_vectors) > 0.85).astype(np.float32)
            
            # Add some spatial correlation (neighbors are likely to be emphasized together)
            for i in range(n_vectors):
                if mask[i] == 1 and i < n_vectors - 1:
                    if np.random.random() < 0.4:  # 40% chance neighbor is also emphasized
                        mask[i+1] = 1
            
            vectors_list.append(vectors)
            masks_list.append(mask)
        
        return EmphasisDataset(vectors_list, masks_list, n_fixed)


# ============================================================================
# 7. COMPLETE PIPELINE EXAMPLE
# ============================================================================

def create_model_pipeline(model_type='transformer', n_fixed=120, device=None):
    """Create a complete model pipeline."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model based on type
    if model_type == 'transformer':
        model = TransformerEmphasisPredictor(n_fixed=n_fixed)
    elif model_type == 'autoencoder':
        model = AutoencoderEmphasisPredictor(n_fixed=n_fixed)
    elif model_type == 'lstm':
        model = LSTMBidirectionalPredictor(n_fixed=n_fixed)
    elif model_type == 'conv':
        model = ConvEmphasisPredictor(n_fixed=n_fixed)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    
    # Create trainer
    trainer = EmphasisModelTrainer(model, device)
    
    emphasis_logger.info("Model pipeline created.")
    emphasis_logger.info(f"  Model Type: {model_type}")
    emphasis_logger.info("====================================")
    
    # Function to apply emphasis with given k
    def apply_emphasis_pipeline(sequences_list, id2embed, vector_dim=300, k=2.0, threshold=0.5, batch_size=32):
        model.eval()
        vectors_list = []
        
        # If id2embed is None assume sequences_list already contains vector arrays
        if id2embed is None:
            raise ValueError("id2embed cannot be None when sequences_list contains token IDs")
        else:
            for seq in sequences_list:
                temp = []
                for idx in seq:
                    if idx in id2embed.keys():
                        temp.append(id2embed[idx])
                    else:
                        temp.append(np.zeros((vector_dim,), dtype=np.float32))
                vectors_list.append(np.asarray(temp, dtype=np.float32))
                
        results = []
        
        # Process in batches
        for i in range(0, len(vectors_list), batch_size):
            batch = vectors_list[i:i+batch_size]
            
            # Convert to tensor
            batch_tensors = []
            for vecs in batch:
                tensor = torch.FloatTensor(vecs).unsqueeze(0)  # Add batch dim
                batch_tensors.append(tensor)
            
            # Pad to same length within batch (for efficiency)
            max_len = max(t.shape[1] for t in batch_tensors)
            padded_batch = []
            
            for tensor in batch_tensors:
                if tensor.shape[1] < max_len:
                    pad_size = max_len - tensor.shape[1]
                    padding = torch.zeros(1, pad_size, vector_dim)
                    padded = torch.cat([tensor, padding], dim=1)
                else:
                    padded = tensor
                padded_batch.append(padded)
            
            # Stack batch
            batch_tensor = torch.cat(padded_batch, dim=0).to(device)
            
            # Predict
            with torch.no_grad():
                if hasattr(model, 'predict_emphasis'):
                    emphasized, binary_mask, probs = model.predict_emphasis(
                        batch_tensor, threshold=threshold, k=k
                    )
                else:
                    # For generic models
                    probs = model(batch_tensor)
                    binary_mask = (probs > threshold).float()
                    
                    # Apply emphasis
                    if hasattr(model, 'preprocess_input'):
                        processed_input = model.preprocess_input(batch_tensor)
                    else:
                        processed_input = batch_tensor
                    
                    mask_expanded = binary_mask.unsqueeze(-1)
                    scaling = 1.0 + (k - 1.0) * mask_expanded
                    emphasized = processed_input * scaling
            
            # Convert to numpy
            emphasized_np = emphasized.cpu().numpy()
            binary_mask_np = binary_mask.cpu().numpy()
            probs_np = probs.cpu().numpy()
            
            # Store results (trim padding if needed)
            for j in range(len(batch)):
                orig_len = len(batch[j])
                results.append((
                    emphasized_np[j][:orig_len],  # Trim padding
                    binary_mask_np[j][:orig_len],  # Trim padding
                    probs_np[j][:orig_len]        # Trim padding
                ))
        
        # If single input, return single result
        if len(results) == 1:
            return results[0]
        
        return results
    
    return model, trainer, apply_emphasis_pipeline


# ============================================================================
# 8. EXAMPLE USAGE
# ============================================================================

def example_usage():
    """Example of how to use the models."""
    print("PyTorch Emphasis Prediction Models")
    print("=" * 50)
    
    # Choose device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Create synthetic dataset
    print("\n1. Creating synthetic dataset...")
    dataset = EmphasisDataset.create_synthetic_data(
        n_samples=100, min_len=80, max_len=150, n_fixed=120
    )
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=16, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=16, shuffle=False
    )
    
    # 2. Create model pipeline
    print("\n2. Creating model...")
    model_type = 'autoencoder'  # Try: 'transformer', 'autoencoder', 'lstm', 'conv'
    model, trainer, pipeline = create_model_pipeline(
        model_type=model_type, 
        n_fixed=120,
        device=device
    )
    
    print(f"Model type: {model_type}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 3. Train model (optional - skip if you just want to test)
    train_model = False
    if train_model:
        print("\n3. Training model...")
        model = trainer.train(
            train_loader, 
            val_loader, 
            epochs=10,  # Use more epochs for real training
            lr=0.001
        )
    
    # 4. Test with sample inputs
    print("\n4. Testing with sample inputs...")
    
    # Create test inputs of different lengths
    test_inputs = [
        np.random.randn(80, 300).astype(np.float32),   # Less than 120
        np.random.randn(120, 300).astype(np.float32),  # Exactly 120
        np.random.randn(150, 300).astype(np.float32)   # More than 120
    ]
    
    for i, test_input in enumerate(test_inputs):
        print(f"\nTest input {i+1}: shape = {test_input.shape}")
        
        # Use the pipeline
        emphasized, mask, probs = pipeline(
            test_input, 
            k=2.0, 
            threshold=0.5
        )
        
        print(f"  Emphasized shape: {emphasized.shape}")
        print(f"  Mask shape: {mask.shape}")
        print(f"  Probabilities shape: {probs.shape}")
        print(f"  Number emphasized: {np.sum(mask)}")
        print(f"  Max probability: {np.max(probs):.3f}")
        print(f"  Min probability: {np.min(probs):.3f}")
    
    # 5. Batch prediction example
    print("\n5. Batch prediction example...")
    
    # Create multiple inputs
    batch_inputs = [
        np.random.randn(np.random.randint(50, 100), 300).astype(np.float32)
        for _ in range(5)
    ]
    
    results = pipeline(
        batch_inputs,
        k=2.0,
        threshold=0.6,
        batch_size=2
    )
    
    for i, (emphasized, mask, probs) in enumerate(results):
        print(f"  Input {i}: original shape = {batch_inputs[i].shape}, "
              f"emphasized shape = {emphasized.shape}, "
              f"emphasized vectors = {int(np.sum(mask))}")
    
    return model, pipeline


# ============================================================================
# 9. QUICK START FUNCTION
# ============================================================================

def create_quick_model(model_type='autoencoder', device=None):
    """
    Quick function to create a ready-to-use model.
    
    Args:
        model_type: 'transformer', 'autoencoder', 'lstm', or 'conv'
        device: 'cuda', 'cpu', or None for auto-detect
    
    Returns:
        model: The PyTorch model
        predict_fn: Function for predictions
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    if model_type == 'transformer':
        model = TransformerEmphasisPredictor(n_fixed=120)
    elif model_type == 'autoencoder':
        model = AutoencoderEmphasisPredictor(n_fixed=120)
    elif model_type == 'lstm':
        model = LSTMBidirectionalPredictor(n_fixed=120)
    elif model_type == 'conv':
        model = ConvEmphasisPredictor(n_fixed=120)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    model.eval()
    
    # Create prediction function
    def predict_emphasis(input_vectors, k=2.0, threshold=0.5):
        """
        Predict emphasis for input vectors.
        
        Args:
            input_vectors: numpy array of shape (n, 300) or list of such arrays
            k: Emphasis multiplier (default: 2.0)
            threshold: Probability threshold (default: 0.5)
        
        Returns:
            If single input: (emphasized_vectors, binary_mask, probabilities)
            If multiple inputs: list of tuples
        """
        # Handle input format
        if isinstance(input_vectors, np.ndarray):
            if len(input_vectors.shape) == 2:
                # Single input
                input_tensor = torch.FloatTensor(input_vectors).unsqueeze(0).to(device)
                single_input = True
            else:
                raise ValueError("Input should be 2D (n, 300) or list of 2D arrays")
        elif isinstance(input_vectors, list):
            # Batch input
            input_tensor = torch.stack([
                torch.FloatTensor(arr).unsqueeze(0) for arr in input_vectors
            ]).to(device)
            single_input = False
        else:
            raise ValueError("Input should be numpy array or list of numpy arrays")
        
        with torch.no_grad():
            # Get predictions
            if hasattr(model, 'predict_emphasis'):
                emphasized, binary_mask, probs = model.predict_emphasis(
                    input_tensor, threshold=threshold, k=k
                )
            else:
                # Generic fallback
                probs = model(input_tensor)
                binary_mask = (probs > threshold).float()
                
                if hasattr(model, 'preprocess_input'):
                    processed_input = model.preprocess_input(input_tensor)
                else:
                    processed_input = input_tensor
                
                mask_expanded = binary_mask.unsqueeze(-1)
                scaling = 1.0 + (k - 1.0) * mask_expanded
                emphasized = processed_input * scaling
        
        # Convert to numpy
        emphasized = emphasized.cpu().numpy()
        binary_mask = binary_mask.cpu().numpy()
        probs = probs.cpu().numpy()
        
        if single_input:
            # Remove batch dimension
            return emphasized[0], binary_mask[0], probs[0]
        else:
            return list(zip(emphasized, binary_mask, probs))
    
    return model, predict_emphasis


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("PyTorch Emphasis Prediction Models")
    print("=" * 60)
    
    # Quick demo
    try:
        model, predict_fn = create_quick_model('autoencoder')
        print(f"Model created successfully on {next(model.parameters()).device}")
        
        # Test with a sample
        sample_input = np.random.randn(100, 300).astype(np.float32)
        emphasized, mask, probs = predict_fn(sample_input, k=2.0, threshold=0.5)
        
        print(f"\nSample input shape: {sample_input.shape}")
        print(f"Output shape: {emphasized.shape}")
        print(f"Number of emphasized vectors: {np.sum(mask)}")
        print(f"Emphasis applied successfully!")
        
    except Exception as e:
        print(f"Error during demo: {e}")
    
    print("\n" + "=" * 60)
    print("Available models:")
    print("1. TransformerEmphasisPredictor - Most powerful, uses self-attention")
    print("2. AutoencoderEmphasisPredictor - Autoencoder with mask prediction")
    print("3. LSTMBidirectionalPredictor - Sequential model with LSTM")
    print("4. ConvEmphasisPredictor - Lightweight convolutional model")
    print("\nUsage:")
    print("  model, predict_fn = create_quick_model('autoencoder')")
    print("  emphasized, mask, probs = predict_fn(input_vectors, k=2.0, threshold=0.5)")