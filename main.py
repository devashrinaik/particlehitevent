# import argparse
# from Data.dataset import PixelClusterDataset
# from Data.visualize_dataset import plot_sample_distribution, plot_target_distribution, plot_all_target_distributions_across_files
# import sys
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# def main():
#     parser = argparse.ArgumentParser(description="Lazy-loaded pixel dataset visualizer")
#     parser.add_argument("--data_dir", type=str, default="training_data")
#     parser.add_argument("--visualize", action="store_true")
#     parser.add_argument("--file_index", type=int, default=0)
#     parser.add_argument("--save_dir", type=str, help="Directory to save plots (optional)")
#     parser.add_argument("--plot_all_targets", action="store_true", help="Plot all 9 target variables")



#     args = parser.parse_args()

#     print("[DEBUG] Loading dataset...")
#     dataset = PixelClusterDataset(args.data_dir)

#     # if args.visualize:
#     #     print("[DEBUG] Visualizing cluster distribution...")
#     #     plot_sample_distribution(dataset)
#     #     print("[DEBUG] Visualizing target values for file...")
#     #     plot_target_distribution(dataset, file_index=args.file_index)

#     if args.visualize:
#         print("[DEBUG] Visualizing cluster distribution...")
#         plot_sample_distribution(dataset, save_dir=args.save_dir)
#         print("[DEBUG] Visualizing target values for file...")
#         plot_target_distribution(dataset, file_index=args.file_index, save_dir=args.save_dir)

#     if args.plot_all_targets:
#         print("[DEBUG] Plotting all target variables...")
#         plot_all_target_distributions_across_files(dataset, num_files=10, save_dir=args.save_dir)

# if __name__ == "__main__":
#     main()


# import os
# import argparse
# import torch
# import matplotlib.pyplot as plt
# import numpy as np
# from torch.utils.data import DataLoader, random_split
# from tqdm import tqdm
# import random
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from Data.dataset import PixelClusterDataset
# from Data.download import download_and_extract_data
# from Models.s4network import S4PredictionNetwork
# from Models.training import train_model, test_model, save_predictions

# def create_url_file(start_num, end_num, base_url, output_file="file_links.txt"):
#     """Create a file with download URLs for the dataset."""
#     with open(output_file, "w") as f:
#         for i in range(start_num, end_num + 1):
#             url = f"{base_url}{i}.out.gz"
#             f.write(url + "\n")
#     print(f"{output_file} has been created with all URLs.")

# def plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies, output_dir="results"):
#     """Plot training and testing metrics over epochs."""
#     os.makedirs(output_dir, exist_ok=True)

#     # Plot losses
#     plt.figure(figsize=(10, 5))
#     plt.plot(train_losses, label='Training Loss')
#     plt.plot(test_losses, label='Testing Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title('Loss over Epochs')
#     plt.legend()
#     plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
#     plt.close()

#     # Plot accuracies
#     plt.figure(figsize=(10, 5))
#     plt.plot(train_accuracies, label='Training Accuracy')
#     plt.plot(test_accuracies, label='Testing Accuracy')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy')
#     plt.title('Accuracy over Epochs')
#     plt.legend()
#     plt.savefig(os.path.join(output_dir, 'accuracy_plot.png'))
#     plt.close()

# def main(args):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     os.makedirs(args.data_dir, exist_ok=True)
#     os.makedirs(args.output_dir, exist_ok=True)

#     if args.create_url_file:
#         create_url_file(args.start_num, args.end_num, args.base_url, args.url_file)

#     if args.download_data:
#         print("Downloading and extracting data...")
#         download_and_extract_data(
#             start_num=args.start_num,
#             end_num=args.end_num,
#             base_url=args.base_url,
#             output_dir=args.data_dir
#         )

#     print("Loading dataset...")
#     dataset = PixelClusterDataset(args.data_dir, sequence_length=args.sequence_length, device=device)

#     print(f"Total events found: {len(dataset)}")
#     print("Clusters per file:")
#     file_event_counts = {}
#     for file_path, _ in dataset.index_map:
#         fname = os.path.basename(file_path)
#         file_event_counts[fname] = file_event_counts.get(fname, 0) + 1
#     for fname, count in file_event_counts.items():
#         print(f"  {fname}: {count} clusters")

#     train_size = int(args.train_split * len(dataset))
#     test_size = len(dataset) - train_size
#     train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

#     train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
#     test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

#     print("Inspecting one training batch...")
#     for inputs, targets in train_loader:
#         print(f"Training batch - Inputs shape: {inputs.shape}")
#         print(f"Training batch - Targets shape: {targets.shape}")
#         break

#     print("Initializing model...")
#     model = S4PredictionNetwork(args.hidden_dim, args.output_dim, height=args.height, width=args.width)
#     model.to(device)

#     print("Starting training...")
#     train_losses, test_losses, train_accuracies, test_accuracies = train_model(
#         model, 
#         train_loader, 
#         test_loader,
#         num_epochs=args.epochs, 
#         learning_rate=args.learning_rate,
#         max_grad_norm=args.max_grad_norm,
#         device=device
#     )

#     print("Evaluating final test loss and accuracy...")
#     final_test_loss, final_test_accuracy = test_model(model, test_loader, device, return_accuracy=True)
#     print(f"Final Test Loss (MSE): {final_test_loss:.4f}")
#     print(f"Final Test Accuracy: {final_test_accuracy:.4f}")

#     print("Saving predictions...")
#     save_predictions(model, test_loader, os.path.join(args.output_dir, "predictions.csv"), device)

#     print("Plotting training curves...")
#     plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies, args.output_dir)

#     print("Saving model...")
#     torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))
#     print(f"Model saved to {os.path.join(args.output_dir, 'model.pt')}")

#     print("Saving state-space matrices...")
#     model.s4_layer.save_matrices()

#     try:
#         from utils.visualize_matrices import plot_all_matrices
#         plot_all_matrices()
#     except ImportError:
#         print("Visualization module not found. Skipping matrix plots.")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Pixel Cluster Prediction with S4 Network")

#     parser.add_argument("--data_dir", type=str, default="training_data", help="Directory containing training data")
#     parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")
#     parser.add_argument("--create_url_file", action="store_true", help="Create URL file for data download")
#     parser.add_argument("--url_file", type=str, default="file_links.txt", help="File to store URLs")
#     parser.add_argument("--download_data", action="store_true", help="Download and extract data files")
#     parser.add_argument("--start_num", type=int, default=16401, help="Start number for data download")
#     parser.add_argument("--end_num", type=int, default=16480, help="End number for data download")
#     parser.add_argument("--base_url", type=str, default="https://cernbox.cern.ch/remote.php/dav/public-files/wJhzxMJq6gmbhE7/pixel_clusters_d", help="Base URL for data download")

#     parser.add_argument("--sequence_length", type=int, default=100, help="Sequence length for input data")
#     parser.add_argument("--hidden_dim", type=int, default=500, help="Hidden dimension for model")
#     parser.add_argument("--output_dim", type=int, default=9, help="Output dimension for model")
#     parser.add_argument("--height", type=int, default=13, help="Height of input image")
#     parser.add_argument("--width", type=int, default=21, help="Width of input image")

#     parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
#     parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
#     parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimizer")
#     parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm for clipping")
#     parser.add_argument("--train_split", type=float, default=0.8, help="Fraction of data to use for training")

#     args = parser.parse_args()
#     main(args)


# import os
# import argparse
# import torch
# from torch.utils.data import DataLoader, random_split

# from Data.dataset import PixelClusterDataset
# from Models.s4network import S4PredictionNetwork
# from Models.training import (
#     train_model, test_model,
#     save_predictions, plot_training_curves
# )

# if __name__ == "__main__":
#     torch.multiprocessing.set_start_method("spawn", force=True)

#     parser = argparse.ArgumentParser(description="Train S4 on Pixel Cluster Dataset")
#     parser.add_argument("--data_dir", type=str, default="training_data")
#     parser.add_argument("--output_dir", type=str, default="results")
#     parser.add_argument("--sequence_length", type=int, default=100)
#     parser.add_argument("--hidden_dim", type=int, default=500)
#     parser.add_argument("--output_dim", type=int, default=9)
#     parser.add_argument("--height", type=int, default=13)
#     parser.add_argument("--width", type=int, default=21)
#     parser.add_argument("--batch_size", type=int, default=32)
#     parser.add_argument("--epochs", type=int, default=100)
#     parser.add_argument("--learning_rate", type=float, default=0.001)
#     parser.add_argument("--max_grad_norm", type=float, default=1.0)
#     args = parser.parse_args()

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"[INFO] Using device: {device}")

#     os.makedirs(args.output_dir, exist_ok=True)

#     print("[INFO] Loading dataset...")
#     dataset = PixelClusterDataset(
#         data_dir=args.data_dir,
#         sequence_length=args.sequence_length,
#         device=device
#     )

#     total_len = len(dataset)
#     train_len = int(0.7 * total_len)
#     val_len = int(0.1 * total_len)
#     test_len = total_len - train_len - val_len
#     train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len])

#     train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
#     val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
#     test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

#     print("[INFO] Initializing model...")
#     model = S4PredictionNetwork(
#         hidden_output_dim=args.hidden_dim,
#         final_output_dim=args.output_dim,
#         height=args.height,
#         width=args.width
#     ).to(device)

#     print("[INFO] Starting training...")
#     train_losses, val_losses, train_accs, val_accs = train_model(
#         model=model,
#         train_loader=train_loader,
#         val_loader=val_loader,
#         num_epochs=args.epochs,
#         learning_rate=args.learning_rate,
#         max_grad_norm=args.max_grad_norm,
#         device=device
#     )

#     print("[INFO] Evaluating on test set...")
#     test_loss, test_acc = test_model(model, test_loader, device, return_accuracy=True)
#     print(f"[RESULT] Final Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

#     print("[INFO] Saving predictions and model...")
#     save_predictions(model, test_loader, os.path.join(args.output_dir, "test_predictions.csv"), device)
#     torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))
#     model.s4_layer.save_matrices(output_dir=os.path.join(args.output_dir, "matrices"))

#     print("[INFO] Plotting training curves...")
#     plot_training_curves(train_losses, val_losses, train_accs, val_accs, args.output_dir)

#     print("[DONE] Training complete.")

########################################################
# # main.py before moving ymodule to input 
# import logging
# from Models.training import train_one_epoch, evaluate, plot_predictions, plot_loss_curve
# from Models.visualize_matrices import save_and_plot_matrices
# import seaborn as sns
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import matplotlib.pyplot as plt
# import numpy as np
# import logging
# import os
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from torch.utils.data import DataLoader, random_split
# from Data.dataset import PixelClusterDataset
# from Models.s4network import S4PredictionNetwork

# logging.basicConfig(
#     level=logging.DEBUG,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     handlers=[
#         logging.FileHandler("logs/debug.log", mode='w'),
#         logging.StreamHandler(sys.stdout)
#     ]
# )

# def main():
#     logging.info("Starting main training pipeline...")
#     data_dir = "preprocessed_data"
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     logging.info(f"Using device: {device}")

#     full_dataset = PixelClusterDataset(data_dir, device=device)
#     total_size = len(full_dataset)
#     train_size = int(0.7 * total_size)
#     val_size = int(0.1 * total_size)
#     test_size = total_size - train_size - val_size
#     logging.info(f"Dataset split - Train: {train_size}, Val: {val_size}, Test: {test_size}")

#     train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
#     train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=64)
#     test_loader = DataLoader(test_dataset, batch_size=64)

#     model = S4PredictionNetwork(hidden_output_dim=128, final_output_dim=9).to(device)
#     logging.info("Model initialized.")

#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#     logging.info("Loss function and optimizer configured.")

#     train_losses = []
#     val_losses = []

#     for epoch in range(1, 6):
#         logging.info(f"Epoch {epoch} starting...")
#         train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
#         val_loss, val_preds, val_targets = evaluate(model, val_loader, criterion, device)
#         plot_predictions(val_preds, val_targets, save_path=f"logs/val_pred_vs_target_epoch{epoch}.png")
#         train_losses.append(train_loss)
#         val_losses.append(val_loss)

#     plot_loss_curve(train_losses, val_losses, save_path="logs/loss_curve.png")

#     logging.info("Evaluating on test set...")
#     test_loss, test_preds, test_targets = evaluate(model, test_loader, criterion, device)
#     plot_predictions(test_preds, test_targets, save_path="logs/test_pred_vs_target.png")
#     logging.info(f"Final Test Loss: {test_loss:.4f}")

#     torch.save(model.state_dict(), "logs/final_model.pt")
#     logging.info("Training complete. Model saved.")

#     # Save final model
#     torch.save(model.state_dict(), "logs/final_model.pt")
#     logging.info("Final model saved to logs/final_model.pt")

#     # Save A, B, C matrices and visualize them
#     save_and_plot_matrices(model, output_dir="logs/matrices")
#     logging.info("Saved A, B, C matrices and visualizations to logs/matrices")

#     # === Matrix Inspection: Print summaries to log ===
#     with torch.no_grad():
#         s4 = model.s4_layer
#         A_masked = (s4.A * s4.A_mask).detach().cpu()
#         B_masked = (s4.B * s4.B_mask).detach().cpu()
#         C_masked = (s4.C * s4.C_mask).detach().cpu()

#         def log_matrix_stats(name, mat):
#             density = (mat != 0).float().mean().item()
#             mean_val = mat.mean().item()
#             std_val = mat.std().item()
#             logging.info(f"{name} stats — shape: {mat.shape}, density: {density:.4f}, mean: {mean_val:.6f}, std: {std_val:.6f}")

#         log_matrix_stats("A_masked", A_masked)
#         log_matrix_stats("B_masked", B_masked)
#         log_matrix_stats("C_masked", C_masked)

#     logging.info("Training complete.")

# if __name__ == "__main__":
#     main()

########################################################
# main.py after moving ymodule to input 
# import logging
# from Models.training import train_one_epoch, evaluate, plot_predictions, plot_loss_curve
# from Models.visualize_matrices import save_and_plot_matrices
# import seaborn as sns
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from torch.utils.data import DataLoader, random_split
# from Data.dataset import PixelClusterDataset
# from Models.s4network import S4PredictionNetwork

# logging.basicConfig(
#     level=logging.DEBUG,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     handlers=[
#         logging.FileHandler("logs/debug.log", mode='w'),
#         logging.StreamHandler(sys.stdout)
#     ]
# )

# def main():
#     logging.info("Starting main training pipeline...")
#     data_dir = "preprocessed_data"
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     logging.info(f"Using device: {device}")

#     full_dataset = PixelClusterDataset(data_dir, device=device)
#     total_size = len(full_dataset)
#     train_size = int(0.7 * total_size)
#     val_size = int(0.1 * total_size)
#     test_size = total_size - train_size - val_size
#     logging.info(f"Dataset split - Train: {train_size}, Val: {val_size}, Test: {test_size}")

#     train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
#     train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=64)
#     test_loader = DataLoader(test_dataset, batch_size=64)

#     model = S4PredictionNetwork(
#         input_dim=274,  # Updated for promoted target
#         hidden_output_dim=128,
#         final_output_dim=8  # One target moved to input
#     ).to(device)
#     logging.info("Model initialized.")

#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#     logging.info("Loss function and optimizer configured.")

#     NUM_EPOCHS = 50
#     PATIENCE = 7
#     best_val_loss = float('inf')
#     patience_counter = 0

#     train_losses = []
#     val_losses = []

#     for epoch in range(1, NUM_EPOCHS + 1):
#         logging.info(f"Epoch {epoch} starting...")
#         train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
#         val_loss, val_preds, val_targets = evaluate(model, val_loader, criterion, device)
#         plot_predictions(val_preds, val_targets, save_path=f"logs/val_pred_vs_target_epoch{epoch}.png")
#         train_losses.append(train_loss)
#         val_losses.append(val_loss)

#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             patience_counter = 0
#             torch.save(model.state_dict(), "logs/best_model.pt")
#             logging.info(f"New best model saved with val_loss {val_loss:.4f}")
#         else:
#             patience_counter += 1
#             logging.info(f"Early stopping counter: {patience_counter}/{PATIENCE}")
#             if patience_counter >= PATIENCE:
#                 logging.info("Early stopping triggered.")
#                 break

#     plot_loss_curve(train_losses, val_losses, save_path="logs/loss_curve.png")

#     logging.info("Evaluating on test set...")
#     model.load_state_dict(torch.load("logs/best_model.pt"))
#     test_loss, test_preds, test_targets = evaluate(model, test_loader, criterion, device)
#     plot_predictions(test_preds, test_targets, save_path="logs/test_pred_vs_target.png")
#     logging.info(f"Final Test Loss: {test_loss:.4f}")

#     torch.save(model.state_dict(), "logs/final_model.pt")
#     logging.info("Final model saved to logs/final_model.pt")

#     save_and_plot_matrices(model, output_dir="logs/matrices")
#     logging.info("Saved A, B, C matrices and visualizations to logs/matrices")

#     with torch.no_grad():
#         s4 = model.s4_layer
#         A_masked = (s4.A * s4.A_mask).detach().cpu()
#         B_masked = (s4.B * s4.B_mask).detach().cpu()
#         C_masked = (s4.C * s4.C_mask).detach().cpu()

#         def log_matrix_stats(name, mat):
#             density = (mat != 0).float().mean().item()
#             mean_val = mat.mean().item()
#             std_val = mat.std().item()
#             logging.info(f"{name} stats — shape: {mat.shape}, density: {density:.4f}, mean: {mean_val:.6f}, std: {std_val:.6f}")

#         log_matrix_stats("A_masked", A_masked)
#         log_matrix_stats("B_masked", B_masked)
#         log_matrix_stats("C_masked", C_masked)

#     logging.info("Training complete.")

# if __name__ == "__main__":
#     main()

########################################################

# main.py after accommodating new input/target dimensions
# main.py (with more evaluations)
# main.py (relevant parts for the fix)
import logging
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Models.training import train_one_epoch, evaluate, plot_predictions, plot_loss_curve, plot_scalar_metric_curves
from Models.visualize_matrices import save_and_plot_matrices
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Import torch.multiprocessing
import torch.multiprocessing as mp

from torch.utils.data import DataLoader, random_split
from Data.dataset import PixelClusterDataset
from Models.s4network import S4PredictionNetwork

ACTUAL_INPUT_DIM_PER_FRAME = 274 
FINAL_TARGET_DIM = 8

# Logging and other global setups can come after setting start method if they don't initialize CUDA
# os.makedirs("logs", exist_ok=True) # This is fine here
# os.makedirs("logs/metrics", exist_ok=True)

# logging.basicConfig(...) # This is also fine here

def main():
    # Ensure logs directory exists before basicConfig tries to write to it
    os.makedirs("logs", exist_ok=True)
    os.makedirs("logs/metrics", exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
        handlers=[
            logging.FileHandler("logs/main_run_eval_enhanced.log", mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("Starting main training pipeline with enhanced evaluations...")
    data_dir = "preprocessed_data"
    # Device selection happens here, which is fine
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # ... (rest of your main function) ...
    full_dataset = PixelClusterDataset(data_dir, device=device)
    total_size = len(full_dataset)
    
    if total_size < 3:
        logging.error(f"Dataset too small ({total_size} samples) to split. Need at least 3.")
        return

    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    if train_size <= 0 or val_size <= 0 or test_size <= 0:
        logging.warning(f"Original dataset splits problem. T:{train_size},V:{val_size},Te:{test_size}. Re-allocating...")
        if total_size >= 3:
            train_size = max(1, int(0.7 * total_size))
            val_size = max(1, int(0.15 * total_size))
            test_size = max(1, total_size - train_size - val_size)
            if test_size <= 0:
                val_size = max(1, total_size - train_size - 1)
                test_size = total_size - train_size - val_size
        if train_size <= 0 or val_size <= 0 or test_size <= 0:
            logging.error(f"Could not make valid data split. Final: T:{train_size},V:{val_size},Te:{test_size}. Aborting.")
            return

    logging.info(f"Dataset split - Total:{total_size}, Train:{train_size}, Val:{val_size}, Test:{test_size}")
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
    
    # DataLoader uses num_workers=2, which triggers multiprocessing
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=device.type == 'cuda')
    val_loader = DataLoader(val_dataset, batch_size=64, num_workers=2, pin_memory=device.type == 'cuda')
    test_loader = DataLoader(test_dataset, batch_size=64, num_workers=2, pin_memory=device.type == 'cuda')

    model = S4PredictionNetwork(
        input_dim=ACTUAL_INPUT_DIM_PER_FRAME,
        hidden_output_dim=128,
        final_output_dim=FINAL_TARGET_DIM
    ).to(device)
    logging.info("Model initialized on device.")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    logging.info("Loss function and optimizer configured.")

    epoch_train_losses = []
    epoch_val_losses = []
    epoch_val_mae_overall = []
    epoch_val_r2_overall = []

    num_epochs = 5

    for epoch in range(1, num_epochs + 1):
        logging.info(f"Epoch {epoch}/{num_epochs} starting...")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device) # Error originates here
        epoch_train_losses.append(train_loss)
        
        logging.info(f"--- Evaluating on Validation Set (Epoch {epoch}) ---")
        val_loss, val_preds, val_targets, val_mae_o, val_r2_o, val_mae_pt, val_r2_pt = evaluate(
            model, val_loader, criterion, device
        )
        epoch_val_losses.append(val_loss)
        epoch_val_mae_overall.append(val_mae_o)
        epoch_val_r2_overall.append(val_r2_o)

        if val_mae_pt is not None and val_r2_pt is not None:
            logging.info(f"Validation MAE (Per Target): {np.array2string(val_mae_pt, formatter={'float_kind':lambda x: '%.4f' % x})}")
            logging.info(f"Validation R2 (Per Target): {np.array2string(val_r2_pt, formatter={'float_kind':lambda x: '%.4f' % x})}")
        
        if val_preds.size > 0 and val_targets.size > 0:
             plot_predictions(val_preds, val_targets, save_path=f"logs/val_pred_vs_target_epoch{epoch}.png")
        else:
            logging.warning(f"Epoch {epoch}: No validation predictions/targets to plot.")
            
    if epoch_train_losses and epoch_val_losses:
        plot_loss_curve(epoch_train_losses, epoch_val_losses, save_path="logs/metrics/loss_curve.png")
    
    metric_curves_to_plot = {}
    if epoch_val_mae_overall:
        metric_curves_to_plot['Validation MAE Overall'] = epoch_val_mae_overall
    if epoch_val_r2_overall:
        metric_curves_to_plot['Validation R2 Overall'] = epoch_val_r2_overall
    
    if metric_curves_to_plot:
        plot_scalar_metric_curves(metric_curves_to_plot, save_dir="logs/metrics")

    logging.info("--- Evaluating on Test Set (Final) ---")
    test_loss, test_preds, test_targets, test_mae_o, test_r2_o, test_mae_pt, test_r2_pt = evaluate(
        model, test_loader, criterion, device
    )
    logging.info(f"Final Test MSE Loss: {test_loss:.4f}")
    logging.info(f"Final Test MAE Overall: {test_mae_o:.4f}")
    logging.info(f"Final Test R2 Overall: {test_r2_o:.4f}")

    if test_mae_pt is not None and test_r2_pt is not None:
        logging.info(f"Final Test MAE (Per Target): {np.array2string(test_mae_pt, formatter={'float_kind':lambda x: '%.4f' % x})}")
        logging.info(f"Final Test R2 (Per Target): {np.array2string(test_r2_pt, formatter={'float_kind':lambda x: '%.4f' % x})}")

    if test_preds.size > 0 and test_targets.size > 0:
        plot_predictions(test_preds, test_targets, save_path="logs/test_pred_vs_target.png")
    else:
        logging.warning("No test predictions/targets to plot for final evaluation.")

    final_model_path = "logs/final_model_state_dict.pt"
    torch.save(model.state_dict(), final_model_path)
    logging.info(f"Final model state_dict saved to {final_model_path}")

    save_and_plot_matrices(model, output_dir="logs/matrices_visualization")
    logging.info("Saved and plotted S4 matrices and input projection using visualize_matrices.py.")

    with torch.no_grad():
        if hasattr(model, 's4_layer'):
            s4 = model.s4_layer
            A_masked = (s4.A * s4.A_mask).detach().cpu() if hasattr(s4,'A') and hasattr(s4,'A_mask') else None
            B_masked = (s4.B * s4.B_mask).detach().cpu() if hasattr(s4,'B') and hasattr(s4,'B_mask') else None
            C_masked = (s4.C * s4.C_mask).detach().cpu() if hasattr(s4,'C') and hasattr(s4,'C_mask') else None

            def log_matrix_stats(name, mat):
                if mat is None or not isinstance(mat, torch.Tensor) or mat.numel() == 0:
                    logging.warning(f"{name} matrix is None, not a Tensor, or empty. Skipping stats.")
                    return
                density = (mat != 0).float().mean().item()
                mean_val = mat.mean().item()
                std_val = mat.std().item()
                logging.info(f"{name} stats — shape: {mat.shape}, density: {density:.4f}, mean: {mean_val:.6f}, std: {std_val:.6f}")

            log_matrix_stats("S4_A_masked", A_masked)
            log_matrix_stats("S4_B_masked_internal", B_masked)
            log_matrix_stats("S4_C_masked", C_masked)
            
            if hasattr(s4, 'input_projection'):
                input_proj_weight = s4.input_projection.weight.detach().cpu()
                log_matrix_stats("S4_input_projection_weight", input_proj_weight)
                if s4.input_projection.bias is not None:
                     input_proj_bias = s4.input_projection.bias.detach().cpu()
                     log_matrix_stats("S4_input_projection_bias", input_proj_bias)
        else:
            logging.warning("Model does not have an 's4_layer' attribute for matrix inspection.")

    logging.info("Training complete.")


if __name__ == "__main__":
    # Set the start method for multiprocessing BEFORE any CUDA initialization
    # or DataLoader with num_workers > 0 is created.
    # 'spawn' is generally recommended for CUDA compatibility.
    if sys.platform.startswith('linux') or sys.platform.startswith('darwin'): # 'fork' is default on Unix
        try:
            # Check if CUDA is available AND you intend to use it with multiple workers
            if torch.cuda.is_available(): # Only set if CUDA is a possibility
                 # Check if `num_workers` in any DataLoader is > 0.
                 # For simplicity, we set it if CUDA is available.
                 # A more robust check would parse args or config for num_workers.
                mp.set_start_method('spawn', force=True)
                print(f"INFO: Multiprocessing start method set to 'spawn' for CUDA compatibility.")
        except RuntimeError as e:
            # This might happen if it's already been set, or in an environment
            # where it cannot be changed (e.g. Jupyter notebook after kernel start).
            print(f"WARNING: Could not set multiprocessing start_method to 'spawn': {e}. "
                  f"If using CUDA with num_workers > 0, this might lead to errors. "
                  f"Current start method: {mp.get_start_method(allow_none=True)}")
    
    main()