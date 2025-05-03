import torch
import torch.nn as nn
import json
import os
import argparse
import sys
import traceback

# Attempt to import the base model class
try:
    from torchmdnet.models.torchmd_et import TorchMD_ET
except ImportError as e:
    print(f"FATAL ERROR: Could not import TorchMD_ET from torchmdnet: {e}", file=sys.stderr)
    print("Please ensure torchmd-net is installed correctly in the environment.", file=sys.stderr)
    sys.exit(1)

# --- Class Definition (Copied from Train_ET_v1_3.py) ---
class CustomTorchMD_ET(TorchMD_ET):
    def __init__(self, num_subunits, subunit_embedding_dim, rmsf_embedding_dim, label_embedding_dim, hidden_channels, num_layers, num_heads, num_rbf, rbf_type, activation, attn_activation, neighbor_embedding, cutoff_lower, cutoff_upper, max_z, max_num_neighbors, *args, **kwargs):
        # Call parent constructor EXACTLY as in the original script
        super().__init__(*args, **kwargs)
        # Initialize custom layers
        self.subunit_embedding = nn.Embedding(num_subunits, subunit_embedding_dim)
        self.rmsf_embedding = nn.Sequential(
            nn.Linear(1, rmsf_embedding_dim),
            nn.ReLU(),
            nn.Linear(rmsf_embedding_dim, rmsf_embedding_dim)
        )
        self.label_embedding = nn.Embedding(2, label_embedding_dim) # Assuming 2 classes for labels (0, 1)

        self.subunit_embedding_dim = subunit_embedding_dim
        self.rmsf_embedding_dim = rmsf_embedding_dim
        self.label_embedding_dim = label_embedding_dim

        # Store hidden_channels if it's needed by the parent class or this class
        # The parent class likely sets self.hidden_channels internally based on its args
        # We might need to access it after super().__init__() if the parent sets it.
        # For the classifier, let's try accessing self.hidden_channels AFTER super() call
        # If super() doesn't set it, we might need to pass hidden_channels explicitly to super()
        # based on the specific torchmd-net version's requirements.

        # Classifier - Input size depends on parent output (self.hidden_channels) + custom features
        # Calculate feature_dim based on how features are concatenated in forward pass
        # From original script's forward pass: feature_dim = max(subunit, rmsf, label embeddings)
        feature_dim = max(subunit_embedding_dim, rmsf_embedding_dim, label_embedding_dim)

        # Get hidden_channels dimension AFTER parent init (assuming parent sets it)
        # Need to handle the case where parent doesn't set it or uses a different name
        # For robustness, let's pass hidden_channels explicitly if needed by the classifier logic
        # Assuming self.hidden_channels is available after super().__init__()
        # If not, replace self.hidden_channels below with the passed hidden_channels argument
        classifier_input_dim = self.hidden_channels + feature_dim

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, self.hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.5), # Assuming dropout was used, adjust if needed
            nn.Linear(self.hidden_channels // 2, 1)
        )

    # --- Forward Pass (Copied from Train_ET_v1_3.py, adapted slightly for clarity) ---
    def forward(self, z, pos, batch, subunit_ids, atom_indices, is_water, rmsf, labels):
        # Call parent forward pass
        # Note: Check what the specific TorchMD_ET version returns.
        # Assuming it returns at least 'x' (node embeddings)
        x, vec, z_ret, pos_ret, batch_ret = super().forward(z, pos, batch)

        # Determine feature dimension based on max of custom embeddings
        feature_dim = max(self.subunit_embedding_dim, self.rmsf_embedding_dim, self.label_embedding_dim)
        features = torch.zeros(z.shape[0], feature_dim, device=z.device)

        non_water_mask = ~is_water
        water_mask = is_water

        # Apply subunit embeddings
        if torch.any(non_water_mask):
            # Clamp subunit IDs to be safe, ensure they are within [0, num_subunits-1]
            valid_subunit_ids = torch.clamp(subunit_ids[non_water_mask], 0, self.subunit_embedding.num_embeddings - 1)
            features[non_water_mask, :self.subunit_embedding_dim] = self.subunit_embedding(valid_subunit_ids)

        # Apply RMSF embeddings
        if torch.any(water_mask):
            rmsf_water = rmsf[water_mask].unsqueeze(1)
            # Handle potential NaNs/Infs in RMSF
            rmsf_water = torch.nan_to_num(rmsf_water, nan=0.0, posinf=10.0, neginf=-10.0)
            rmsf_embeddings = self.rmsf_embedding(rmsf_water)
            features[water_mask, :self.rmsf_embedding_dim] = rmsf_embeddings

        # Apply label embeddings (only if training, but need placeholder size)
        # For inference, we might not have labels, but the concatenation expects the space
        # If labels are None during inference, this part needs adjustment.
        # Assuming labels are provided even if not used by loss (e.g., filled with -1)
        if labels is not None and self.training: # Only use labels during training
             valid_label_mask = (labels[water_mask] >= 0) # Assuming labels are 0 or 1, -1 if invalid
             if torch.any(valid_label_mask):
                  label_input = labels[water_mask][valid_label_mask]
                  label_embeddings = self.label_embedding(label_input)
                  # Place embeddings correctly using the mask
                  features[water_mask][valid_label_mask, :self.label_embedding_dim] = label_embeddings
        # Note: If labels are NOT provided during inference, the feature tensor size
        # might be incorrect for the classifier if label_embedding_dim was included
        # in its input size calculation based on training mode.
        # The classifier definition above uses max(subunit, rmsf, label), so it expects space.

        # Concatenate parent output with custom features
        x = torch.cat([x, features], dim=-1)

        # Apply classifier ONLY to water molecules
        logits = torch.zeros(z.shape[0], 1, device=z.device) # Initialize logits
        if torch.any(water_mask):
             logits[water_mask] = self.classifier(x[water_mask])

        return logits, is_water, labels # Return labels for potential use

# --- Loading Functions (Copied from Train_ET_v1_3.py) ---
def load_config(config_path):
    """Loads configuration JSON."""
    print(f"Attempting to load config from: {config_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print("Config loaded successfully.")
        return config
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON from config file: {config_path} - {e}")
    except Exception as e:
        raise RuntimeError(f"Error loading config {config_path}: {e}")

def load_model(model_path, config, device):
    """Loads the pre-trained CustomTorchMD_ET model."""
    print(f"Attempting to load model weights from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights file not found: {model_path}")

    try:
        checkpoint = torch.load(model_path, map_location=device)
        print("Checkpoint loaded.")
    except Exception as e:
        raise RuntimeError(f"Error loading checkpoint file {model_path}: {e}")

    try:
        # Determine num_subunits - crucial for embedding layer size
        # Defaulting to 4 if not in config, but should ideally match training
        num_subunits = config.get('num_subunits', 4)
        print(f"Initializing model architecture using config (num_subunits={num_subunits})...")

        # Initialize the model using config values EXACTLY as in the original script
        # Pass only the arguments explicitly defined in CustomTorchMD_ET's __init__
        # Any extra args in config will be captured by **kwargs if present in signature,
        # but the parent call `super().__init__(*args, **kwargs)` is the key part.
        model = CustomTorchMD_ET(
            num_subunits=num_subunits,
            subunit_embedding_dim=config['subunit_embedding_dim'],
            rmsf_embedding_dim=config['rmsf_embedding_dim'],
            label_embedding_dim=config['label_embedding_dim'],
            hidden_channels=config['hidden_channels'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            num_rbf=config['num_rbf'],
            rbf_type=config['rbf_type'],
            activation=config['activation'],
            attn_activation=config['attn_activation'],
            neighbor_embedding=config['neighbor_embedding'],
            cutoff_lower=config['cutoff_lower'],
            cutoff_upper=config['cutoff_upper'],
            max_z=config['max_z'],
            max_num_neighbors=config['max_num_neighbors']
            # DO NOT pass dtype=torch.float32 here if parent doesn't expect it
            # DO NOT pass embedding_dimension, attn_dropout, dropout etc. explicitly here
            # Let *args, **kwargs in __init__ handle extras if needed by parent
        ).to(device)
        print("Model architecture initialized.")

        # Load the state dict
        # Check if state_dict is nested under 'model_state_dict' key
        state_dict_to_load = checkpoint
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict_to_load = checkpoint['model_state_dict']
            print("Loading state_dict from 'model_state_dict' key in checkpoint.")
        else:
            print("Loading state_dict directly from checkpoint.")

        # Load the weights
        model.load_state_dict(state_dict_to_load) # Set strict=True by default
        print("Model weights loaded successfully into architecture.")

        model.eval() # Set to evaluation mode
        print("Model set to evaluation mode.")
        return model

    except FileNotFoundError:
         raise # Re-raise specific error
    except KeyError as e:
         raise RuntimeError(f"Missing key in config file needed for model init: {e}")
    except Exception as e:
         # Catch state_dict loading errors here
         print("\n--- ERROR during model.load_state_dict() ---")
         print(f"Error Type: {type(e).__name__}")
         print(f"Error Message: {e}")
         print("This usually indicates a mismatch between the model architecture defined")
         print("by the config file and the architecture saved in the weights file.")
         print("Check 'hidden_channels', 'num_layers', and other architectural parameters.")
         print("--------------------------------------------\n")
         # Re-raise the exception to stop the script
         raise RuntimeError(f"Error loading model state_dict from {model_path}: {e}") from e


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal script to test loading the TorchMD-ET model.")
    parser.add_argument("--config", required=True, help="Path to the JSON configuration file.")
    parser.add_argument("--model", required=True, help="Path to the trained model weights (.pth) file.")
    args = parser.parse_args()

    print("--- Starting Model Load Test ---")
    print(f"Config File: {args.config}")
    print(f"Model File: {args.model}")

    try:
        # 1. Load Config
        config = load_config(args.config)

        # 2. Set Device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # 3. Load Model
        model = load_model(args.model, config, device)

        print("\n--- SUCCESS: Model loaded successfully! ---")
        # Optionally print model summary
        # print(model)

    except FileNotFoundError as e:
        print(f"\n--- FAILURE ---")
        print(f"Error: {e}")
        print("Please ensure the specified config and model files exist.")
        sys.exit(1)
    except (ValueError, RuntimeError, TypeError) as e: # Catch config or loading errors
        print(f"\n--- FAILURE ---")
        print(f"Error: {e}")
        # traceback.print_exc() # Print full traceback for debugging
        sys.exit(1)
    except Exception as e: # Catch any other unexpected errors
        print(f"\n--- FAILURE ---")
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()
        sys.exit(1)

    print("--- Test Finished ---")
