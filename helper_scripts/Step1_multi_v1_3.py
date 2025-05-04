# Step1 Muti Ver 1.3 (before rmsf terms splitting)

import os
import random
import json
import argparse
import logging
import torch
import numpy as np
from torch.utils.data import ConcatDataset, DataLoader
from torch_geometric.data import Batch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import uuid
from sklearn.metrics import f1_score, roc_auc_score

# Import necessary functions from Train_ET.py
from Train_ET_v1_3 import (
    load_dataset,
    prepare_data,
    CustomTorchMD_ET,
    setup_logging
)

def custom_collate(batch):
    if isinstance(batch[0], torch.Tensor):
        return torch.stack(batch, dim=0)
    elif isinstance(batch[0], float):
        return torch.tensor(batch, dtype=torch.float)
    elif isinstance(batch[0], int):
        return torch.tensor(batch, dtype=torch.long)
    elif isinstance(batch[0], (list, tuple)):
        return [custom_collate(samples) for samples in zip(*batch)]
    elif isinstance(batch[0], dict):
        return {key: custom_collate([d[key] for d in batch]) for key in batch[0]}
    else:
        return Batch.from_data_list(batch)

def train_epoch(model, train_loader, optimizer, device, accumulation_steps):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, batch in enumerate(train_loader):
        batch = batch.to(device)
        logits, is_water, labels = model(batch.z, batch.pos, batch.batch, batch.subunit_ids, 
                                         batch.atom_indices, batch.is_water, batch.rmsf, batch.labels)
        loss = model.compute_loss(logits, is_water, labels)

        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()

        # Compute accuracy
        water_mask = is_water.bool()
        water_logits = logits[water_mask].squeeze(-1)
        water_labels = labels[water_mask]
        valid_label_mask = water_labels != -1

        predictions = (water_logits[valid_label_mask] > 0).float()
        correct += (predictions == water_labels[valid_label_mask]).sum().item()
        total += valid_label_mask.sum().item()

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total if total > 0 else 0

    return avg_loss, accuracy

def validate(model, val_loader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            logits, is_water, labels = model(batch.z, batch.pos, batch.batch, batch.subunit_ids, 
                                             batch.atom_indices, batch.is_water, batch.rmsf, batch.labels)
            loss = model.compute_loss(logits, is_water, labels)
            
            total_loss += loss.item()
            
            water_mask = is_water.bool()
            water_logits = logits[water_mask].squeeze(-1)
            water_labels = labels[water_mask]
            valid_label_mask = water_labels != -1
            
            predictions = (water_logits[valid_label_mask] > 0).float()
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(water_labels[valid_label_mask].cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    avg_loss = total_loss / len(val_loader)
    accuracy = (all_preds == all_labels).mean()
    f1 = f1_score(all_labels, all_preds, average='binary')
    try:
        auc_roc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        auc_roc = float('nan')  # In case of single class prediction
    
    return avg_loss, accuracy, f1, auc_roc

def load_multiple_datasets(config, datasets):
    all_data = []
    for dataset_number in datasets:
        universe, labels_dict, frames_mask = load_dataset(
            config,
            dataset_number,
            config['start_frame'],
            config['end_frame'],
            config['stride']
        )
        data_list = prepare_data(
        universe, 
        labels_dict, 
        frames_mask, 
        sequence=config['sequence'],
        n_upstream=config['n_upstream'],
        n_downstream=config['n_downstream'],
        cylinder_radius=config['cylinder_radius'],
        initial_height=config['initial_height'],
        min_waters=config['min_waters'],
        window_size=config['window_size']
    )

        all_data.extend(data_list)
    return all_data

def create_synthetic_validation_set(all_data, config):
    validation_set = []
    train_size = int(0.8 * len(all_data))
    validation_set = all_data[train_size:]
    
    # Load and add data from datasets 9 and 10
    for dataset_number in [9, 10]:
        universe, labels_dict, frames_mask = load_dataset(
            config,
            dataset_number,
            config['start_frame'],
            config['end_frame'],
            config['stride']
        )
        data_list = prepare_data(
        universe, 
        labels_dict, 
        frames_mask, 
        sequence=config['sequence'],
        n_upstream=config['n_upstream'],
        n_downstream=config['n_downstream'],
        cylinder_radius=config['cylinder_radius'],
        initial_height=config['initial_height'],
        min_waters=config['min_waters'],
        window_size=config['window_size']
        )

        num_val_frames = min(100, len(data_list))  # Take 100 frames or all if less than 100
        validation_set.extend(random.sample(data_list, num_val_frames))
    
    random.shuffle(validation_set)
    return validation_set

def main(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Generate a unique run ID
    run_id = str(uuid.uuid4())
    
    # Create a unique output directory for this run
    output_dir = os.path.join(config['output_dir'], run_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # Update the config with the new output directory
    config['output_dir'] = output_dir
    
    # Save the updated configuration to the new output directory
    config_output_path = os.path.join(output_dir, 'config.json')
    with open(config_output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Setup logging
    setup_logging(output_dir)
    logging.info(f"Starting run {run_id} in directory: {output_dir}")
    logging.info(f"Configuration saved to: {config_output_path}")
    
    # Set device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Set random seeds for reproducibility
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Load datasets 1 to 8
    all_data = load_multiple_datasets(config, range(1, 9))

    # Create training and validation sets
    train_size = int(0.8 * len(all_data))
    train_data = all_data[:train_size]
    validation_set = create_synthetic_validation_set(all_data[train_size:], config)

    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(validation_set, batch_size=config['batch_size'], shuffle=False, collate_fn=custom_collate)

    # Determine num_subunits from the sample data
    num_subunits = max(data.subunit_ids.max().item() for data in train_data) + 1

    # Initialize model
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
        max_num_neighbors=config['max_num_neighbors'],
        dtype=torch.float32
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)
    # Removed verbose parameter due to PyTorch version compatibility

    patience = config['patience']
    min_delta = 0.001
    patience_counter = 0
    best_val_loss = float('inf')

    for epoch in range(config['num_epochs']):
        logging.info(f"Epoch {epoch+1}/{config['num_epochs']}")
        train_loss, train_accuracy = train_epoch(model, train_loader, optimizer, device, config['accumulation_steps'])
        val_loss, val_accuracy, val_f1, val_auc_roc = validate(model, val_loader, device)

        logging.info(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        logging.info(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}, Val AUC-ROC: {val_auc_roc:.4f}")

        scheduler.step(val_loss)

        # Save checkpoint at specified interval
        if (epoch + 1) % config['checkpoint_interval'] == 0:
            checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'val_f1': val_f1,
                'val_auc_roc': val_auc_roc,
            }, checkpoint_path)
            logging.info(f"Saved checkpoint at epoch {epoch+1}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'val_f1': val_f1,
                'val_auc_roc': val_auc_roc,
            }, os.path.join(output_dir, 'best_model.pth'))
            logging.info("New best model saved!")

        # Early stopping check
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"Early stopping triggered after {epoch+1} epochs")
                break

        # Log the difference between train and val metrics to monitor overfitting
        logging.info(f"Train-Val Loss Diff: {train_loss - val_loss:.4f}")
        logging.info(f"Train-Val Accuracy Diff: {train_accuracy - val_accuracy:.4f}")

    logging.info("Training completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ET model on multiple datasets")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()
    main(args.config)