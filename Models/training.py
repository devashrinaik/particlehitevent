# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# import pandas as pd
# import os
# from tqdm import tqdm
# import matplotlib.pyplot as plt

# def initialize_weights(m):
#     """Initialize model weights."""
#     if isinstance(m, nn.Linear):
#         nn.init.xavier_uniform_(m.weight)
#         if m.bias is not None:
#             nn.init.zeros_(m.bias)

# def calculate_accuracy(predictions, targets, threshold=0.1):
#     """Calculate accuracy based on a threshold."""
#     # Define accuracy as prediction within threshold of target
#     with torch.no_grad():
#         diff = torch.abs(predictions - targets)
#         correct = (diff <= threshold).float().mean(dim=1)
#         return correct.mean().item()
    
# def plot_training_curves(train_losses, test_losses, train_accuracies, test_accuracies, output_dir="results"):
#     """Plot training loss and accuracy curves."""
#     os.makedirs(output_dir, exist_ok=True)

#     # Plot loss
#     plt.figure()
#     plt.plot(train_losses, label="Train Loss")
#     plt.plot(test_losses, label="Validation Loss")
#     plt.xlabel("Epoch")
#     plt.ylabel("MSE Loss")
#     plt.title("Loss Curve")
#     plt.legend()
#     plt.savefig(os.path.join(output_dir, "loss_curve.png"))
#     plt.close()

#     # Plot accuracy
#     plt.figure()
#     plt.plot(train_accuracies, label="Train Accuracy")
#     plt.plot(test_accuracies, label="Validation Accuracy")
#     plt.xlabel("Epoch")
#     plt.ylabel("Accuracy")
#     plt.title("Accuracy Curve")
#     plt.legend()
#     plt.savefig(os.path.join(output_dir, "accuracy_curve.png"))
#     plt.close()

#     print(f"[INFO] Plots saved to: {output_dir}")

# def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001, max_grad_norm=1.0, device="cuda"):
#     """Train the model and return training metrics."""
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
#     # Initialize model weights
#     model.apply(initialize_weights)
    
#     # Lists to store metrics
#     train_losses = []
#     test_losses = []
#     train_accuracies = []
#     test_accuracies = []
    
#     for epoch in range(num_epochs):
#         # Training phase
#         print(f"Training epoch {epoch+1}...")
#         model.train()
#         epoch_loss = 0
#         epoch_accuracy = 0
#         batches = 0
        
#         for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
#             batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
#             optimizer.zero_grad()
#             predictions = model(batch_x)
#             loss = criterion(predictions, batch_y)
            
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
#             optimizer.step()
            
#             epoch_loss += loss.item()
#             epoch_accuracy += calculate_accuracy(predictions, batch_y)
#             batches += 1
        
#         avg_train_loss = epoch_loss / batches
#         avg_train_accuracy = epoch_accuracy / batches
#         train_losses.append(avg_train_loss)
#         train_accuracies.append(avg_train_accuracy)
        
#         # Testing phase
#         print(f"Testing epoch {epoch+1}...")
#         val_loss, val_accuracy = test_model(model, val_loader, device, return_accuracy=True)
#         test_losses.append(val_loss)
#         test_accuracies.append(val_accuracy)
        
#         # Update learning rate
#         scheduler.step(val_loss)
        
#         print(f"Epoch [{epoch+1}/{num_epochs}], "
#               f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f}, "
#               f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
        
#         # Save checkpoint every 100 epochs
#         if (epoch + 1) % 100 == 0:
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'scheduler_state_dict': scheduler.state_dict(),
#                 'train_loss': avg_train_loss,
#                 'test_loss': val_loss,
#             }, f"results/checkpoint_epoch_{epoch+1}.pt")
    
#     return train_losses, test_losses, train_accuracies, test_accuracies

# def test_model(model, test_loader, device="cuda", return_accuracy=False):
#     """Test the model and return test metrics."""
#     model.eval()
#     criterion = nn.MSELoss()
#     test_loss = 0.0
#     test_accuracy = 0.0
#     num_batches = 0
    
#     with torch.no_grad():
#         for inputs, targets in tqdm(test_loader, desc="Testing"):
#             inputs, targets = inputs.to(device), targets.to(device)
            
#             predictions = model(inputs)
#             loss = criterion(predictions, targets)
            
#             test_loss += loss.item()
            
#             if return_accuracy:
#                 test_accuracy += calculate_accuracy(predictions, targets)
            
#             num_batches += 1
    
#     avg_test_loss = test_loss / num_batches
#     print(f"Test loss: {avg_test_loss:.4f}")

#     if return_accuracy:
#         avg_test_accuracy = test_accuracy / num_batches
#         print(f"Test accuracy: {avg_test_accuracy:.4f}")
#         return avg_test_loss, avg_test_accuracy
    
#     return avg_test_loss

# def save_predictions(model, data_loader, output_file, device="cuda"):
#     """Save model predictions to a CSV file."""
#     model.eval()
    
#     all_predictions = []
#     all_targets = []
    
#     with torch.no_grad():
#         for inputs, targets in tqdm(data_loader, desc="Generating predictions"):
#             inputs, targets = inputs.to(device), targets.to(device)
            
#             predictions = model(inputs)
            
#             all_predictions.extend(predictions.cpu().numpy())
#             all_targets.extend(targets.cpu().numpy())
    
#     # Create DataFrame
#     results = pd.DataFrame()
    
#     # Add targets
#     for i in range(all_targets[0].shape[0]):
#         results[f'target_{i}'] = [t[i] for t in all_targets]
    
#     # Add predictions
#     for i in range(all_predictions[0].shape[0]):
#         results[f'prediction_{i}'] = [p[i] for p in all_predictions]
    
#     # Add error
#     for i in range(all_predictions[0].shape[0]):
#         results[f'error_{i}'] = np.abs(results[f'prediction_{i}'] - results[f'target_{i}'])
    
#     # Save to CSV
#     os.makedirs(os.path.dirname(output_file), exist_ok=True)
#     results.to_csv(output_file, index=False)
#     print(f"Predictions saved to {output_file}")

#     metrics = pd.DataFrame({
#         "epoch": list(range(1, num_epochs + 1)),
#         "train_loss": train_losses,
#         "test_loss": test_losses,
#         "train_accuracy": train_accuracies,
#         "test_accuracy": test_accuracies,
#     })
#     metrics.to_csv("results/training_metrics.csv", index=False)

########################################################
# # training.py
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, random_split
# from Data.dataset import PixelClusterDataset
# from Models.s4network import S4PredictionNetwork
# import logging
# import os
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import matplotlib.pyplot as plt
# import numpy as np

# def train_one_epoch(model, dataloader, criterion, optimizer, device):
#     model.train()
#     total_loss = 0
#     for batch_idx, (x, y) in enumerate(dataloader):
#         optimizer.zero_grad()
#         x, y = x.to(device), y.to(device)
#         output = model(x)
#         loss = criterion(output, y)
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()
#         if batch_idx % 10 == 0:
#             logging.info(f"Batch {batch_idx}, Loss: {loss.item():.4f}")

#     avg_loss = total_loss / len(dataloader)
#     logging.info(f"Average Loss: {avg_loss:.4f}")
#     return avg_loss

# def evaluate(model, dataloader, criterion, device):
#     model.eval()
#     total_loss = 0
#     all_preds = []
#     all_targets = []
#     with torch.no_grad():
#         for x, y in dataloader:
#             x, y = x.to(device), y.to(device)
#             output = model(x)
#             loss = criterion(output, y)
#             total_loss += loss.item()
#             all_preds.append(output.cpu().numpy())
#             all_targets.append(y.cpu().numpy())

#     avg_loss = total_loss / len(dataloader)
#     logging.info(f"Evaluation Loss: {avg_loss:.4f}")
#     return avg_loss, np.concatenate(all_preds), np.concatenate(all_targets)

# def plot_predictions(preds, targets, save_path="logs/pred_vs_target.png"):
#     fig, axs = plt.subplots(3, 3, figsize=(12, 8))
#     axs = axs.flatten()

#     for i in range(preds.shape[1]):
#         axs[i].plot(preds[:100, i], label=f"Pred {i}", linestyle='--')
#         axs[i].plot(targets[:100, i], label=f"True {i}", alpha=0.7)
#         axs[i].set_title(f"Target {i}")
#         axs[i].legend()
#         axs[i].grid(True)

#     plt.tight_layout()
#     plt.savefig(save_path)
#     logging.info(f"Saved prediction subplot figure to {save_path}")
#     logging.info(f"Saved prediction plot to {save_path}")

# def plot_loss_curve(train_losses, val_losses, save_path="logs/loss_curve.png"):
#     plt.figure(figsize=(8, 5))
#     plt.plot(train_losses, label='Train Loss')
#     plt.plot(val_losses, label='Validation Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title('Loss Curve')
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(save_path)
#     logging.info(f"Saved loss curve to {save_path}")

########################################################
# training.py after moving ymodule to input 

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, random_split
# from Data.dataset import PixelClusterDataset
# from Models.s4network import S4PredictionNetwork
# import logging
# import os
# import sys
# import csv
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.metrics import mean_absolute_error, r2_score

# # Constants matching new format
# INPUT_DIM = 274  # 273 original + 1 promoted
# OUTPUT_DIM = 8   # 9 original - 1 promoted

# def train_one_epoch(model, dataloader, criterion, optimizer, device):
#     model.train()
#     total_loss = 0
#     for batch_idx, (x, y) in enumerate(dataloader):
#         optimizer.zero_grad()
#         x, y = x.to(device), y.to(device)
#         output = model(x)
#         loss = criterion(output, y)
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()
#         if batch_idx % 10 == 0:
#             logging.info(f"Batch {batch_idx}, Loss: {loss.item():.4f}")

#     avg_loss = total_loss / len(dataloader)
#     logging.info(f"Average Loss: {avg_loss:.4f}")
#     return avg_loss

# def evaluate(model, dataloader, criterion, device, save_csv_path="logs/metrics_summary.csv"):
#     model.eval()
#     total_loss = 0
#     all_preds = []
#     all_targets = []
#     with torch.no_grad():
#         for x, y in dataloader:
#             x, y = x.to(device), y.to(device)
#             output = model(x)
#             loss = criterion(output, y)
#             total_loss += loss.item()
#             all_preds.append(output.cpu().numpy())
#             all_targets.append(y.cpu().numpy())

#     avg_loss = total_loss / len(dataloader)
#     preds = np.concatenate(all_preds)
#     targets = np.concatenate(all_targets)

#     # Debug: Print shapes
#     logging.debug(f"Predictions shape: {preds.shape}, Targets shape: {targets.shape}")

#     # Additional regression metrics
#     mae = mean_absolute_error(targets, preds)
#     rmse = np.sqrt(((preds - targets) ** 2).mean())
#     r2 = r2_score(targets, preds)

#     logging.info(f"Evaluation Loss (MSE): {avg_loss:.4f}")
#     logging.info(f"MAE:  {mae:.4f}")
#     logging.info(f"RMSE: {rmse:.4f}")
#     logging.info(f"R² Score: {r2:.4f}")

#     # Per-target R^2 scores
#     r2_per_target = []
#     for i in range(targets.shape[1]):
#         target_r2 = r2_score(targets[:, i], preds[:, i])
#         logging.debug(f"Target {i if i < 7 else i+1} - R²: {target_r2:.4f}")
#         r2_per_target.append(target_r2)

#     # Save metrics to CSV
#     os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)
#     with open(save_csv_path, "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["Metric", "Value"])
#         writer.writerow(["MSE", f"{avg_loss:.4f}"])
#         writer.writerow(["MAE", f"{mae:.4f}"])
#         writer.writerow(["RMSE", f"{rmse:.4f}"])
#         writer.writerow(["R2 Overall", f"{r2:.4f}"])
#         for i, r2_val in enumerate(r2_per_target):
#             writer.writerow([f"R2 Target {i if i < 7 else i+1}", f"{r2_val:.4f}"])
#     logging.info(f"Saved evaluation metrics to {save_csv_path}")

#     return avg_loss, preds, targets

# def plot_predictions(preds, targets, save_path="logs/pred_vs_target.png"):
#     num_targets = preds.shape[1]
#     fig_dim = int(np.ceil(np.sqrt(num_targets)))
#     fig, axs = plt.subplots(fig_dim, fig_dim, figsize=(14, 10))
#     axs = axs.flatten()

#     for i in range(num_targets):
#         axs[i].plot(preds[:100, i], label=f"Pred {i}", linestyle='--')
#         axs[i].plot(targets[:100, i], label=f"True {i}", alpha=0.7)
#         axs[i].set_title(f"Target {i if i < 7 else i+1}")  # Skip visual index of promoted one
#         axs[i].legend()
#         axs[i].grid(True)

#     plt.tight_layout()
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     plt.savefig(save_path)
#     logging.info(f"Saved prediction subplot figure to {save_path}")

# def plot_loss_curve(train_losses, val_losses, save_path="logs/loss_curve.png"):
#     plt.figure(figsize=(8, 5))
#     plt.plot(train_losses, label='Train Loss')
#     plt.plot(val_losses, label='Validation Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title('Loss Curve')
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     plt.savefig(save_path)
#     logging.info(f"Saved loss curve to {save_path}")


# Models/training.py (mostly unchanged, robust to new dimensions)
# Models/training.py
import torch
import torch.nn as nn
# import torch.optim as optim # Not directly used in this snippet, but likely in full file
# from torch.utils.data import DataLoader, random_split # Not directly used in this snippet
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Adding project root

# from Data.dataset import PixelClusterDataset # Not directly used in this snippet
# from Models.s4network import S4PredictionNetwork # Not directly used in this snippet

import logging
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score # For additional metrics

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch_idx, (x, y) in enumerate(dataloader):
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if batch_idx % 10 == 0: 
            logging.info(f"Epoch Train Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    logging.info(f"Epoch Train Average Loss: {avg_loss:.4f}")
    return avg_loss

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds_list = []
    all_targets_list = []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output, y)
            total_loss += loss.item()
            all_preds_list.append(output.cpu().numpy())
            all_targets_list.append(y.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    
    if not all_preds_list or not all_targets_list:
        logging.warning("Evaluation produced no predictions or targets.")
        # Return empty/default values to avoid crashing downstream
        empty_preds = np.array([])
        empty_targets = np.array([])
        # Assuming FINAL_TARGET_DIM might be known or passed, or handle downstream
        # For now, let's return NaNs or None for metrics if data is empty
        return avg_loss, empty_preds, empty_targets, np.nan, np.nan, None, None


    all_preds = np.concatenate(all_preds_list)
    all_targets = np.concatenate(all_targets_list)

    # Calculate additional metrics
    # Overall metrics (averaged across all targets and samples)
    mae_overall = mean_absolute_error(all_targets, all_preds, multioutput='uniform_average')
    r2_overall = r2_score(all_targets, all_preds, multioutput='uniform_average')
    
    # Per-target metrics
    # These will return an array of scores, one for each target dimension
    mae_per_target = mean_absolute_error(all_targets, all_preds, multioutput='raw_values')
    r2_per_target = r2_score(all_targets, all_preds, multioutput='raw_values')
    
    logging.info(f"Evaluation MSE Loss: {avg_loss:.4f}")
    logging.info(f"Evaluation MAE Overall: {mae_overall:.4f}")
    logging.info(f"Evaluation R2 Overall: {r2_overall:.4f}")
    # Log per-target if desired, can be verbose:
    # logging.debug(f"Evaluation MAE Per Target: {mae_per_target}")
    # logging.debug(f"Evaluation R2 Per Target: {r2_per_target}")
    
    return avg_loss, all_preds, all_targets, mae_overall, r2_overall, mae_per_target, r2_per_target


def plot_predictions(preds, targets, save_path="logs/pred_vs_target.png"):
    if preds.ndim == 1: # If preds is 1D, reshape to 2D for consistent processing
        preds = preds.reshape(-1, 1)
    if targets.ndim == 1:
        targets = targets.reshape(-1, 1)

    num_features_to_plot = preds.shape[1]
    if num_features_to_plot == 0:
        logging.warning(f"No features to plot in predictions. Skipping plot: {save_path}")
        return

    num_cols = min(3, num_features_to_plot) 
    num_rows = (num_features_to_plot + num_cols - 1) // num_cols

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(4*num_cols, 3*num_rows), squeeze=False)
    axs = axs.flatten()

    for i in range(num_features_to_plot):
        axs[i].plot(preds[:100, i], label=f"Pred Target {i+1}", linestyle='--')
        axs[i].plot(targets[:100, i], label=f"True Target {i+1}", alpha=0.7)
        axs[i].set_title(f"Feature {i+1}")
        axs[i].legend()
        axs[i].grid(True)

    for j in range(num_features_to_plot, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)
    logging.info(f"Saved prediction plot to {save_path}")


def plot_loss_curve(train_losses, val_losses, save_path="logs/loss_curve.png"):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Saved loss curve to {save_path}")

# New function to plot scalar metrics over epochs
def plot_scalar_metric_curves(epoch_metrics_dict, save_dir="logs/metrics"):
    """
    Plots scalar metrics collected over epochs.
    Args:
        epoch_metrics_dict (dict): A dictionary where keys are metric names (e.g., 'Validation MAE')
                                   and values are lists of metric values per epoch.
        save_dir (str): Directory to save the plots.
    """
    os.makedirs(save_dir, exist_ok=True)
    for metric_name, values in epoch_metrics_dict.items():
        if not values or all(np.isnan(v) for v in values):  # Check if list is empty or all NaNs
            logging.warning(f"No valid data to plot for {metric_name}. Skipping.")
            continue
        
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(values) + 1), values, label=metric_name, marker='o', linestyle='-') # Added range for x-axis
        plt.xlabel('Epoch')
        # Attempt to extract a more specific Y-axis label (e.g., MAE, R2)
        simple_metric_name = metric_name.split()[-1] if ' ' in metric_name else metric_name
        plt.ylabel(simple_metric_name)
        plt.title(f'{metric_name} Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.xticks(range(1, len(values) + 1)) # Ensure integer ticks for epochs
        plt.tight_layout()
        
        file_name_suffix = metric_name.lower().replace(' ', '_').replace('[','').replace(']','').replace('(','').replace(')','')
        save_path = os.path.join(save_dir, f"{file_name_suffix}_curve.png")
        plt.savefig(save_path)
        plt.close()
        logging.info(f"Saved {metric_name} curve to {save_path}")