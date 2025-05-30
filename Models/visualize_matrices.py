# # Models/visualize_matrices.py
# import torch
# import matplotlib.pyplot as plt
# import os
# import seaborn as sns

# def plot_matrix(matrix, title, save_path):
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(matrix, cmap="viridis")
#     plt.title(title)
#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.close()

# def save_and_plot_matrices(model, output_dir="results/matrices"):
#     os.makedirs(output_dir, exist_ok=True)

#     # Extract the S4 layer
#     s4 = model.s4_layer

#     # Detach and mask
#     A = (s4.A * s4.A_mask).detach().cpu().numpy()
#     B = (s4.B * s4.B_mask).detach().cpu().numpy()
#     C = (s4.C * s4.C_mask).detach().cpu().numpy()

#     # Save raw matrices
#     torch.save(s4.A.detach().cpu(), os.path.join(output_dir, "A_matrix.pt"))
#     torch.save(s4.B.detach().cpu(), os.path.join(output_dir, "B_matrix.pt"))
#     torch.save(s4.C.detach().cpu(), os.path.join(output_dir, "C_matrix.pt"))

#     # Save plots
#     plot_matrix(A, "A Matrix (Masked)", os.path.join(output_dir, "A_matrix.png"))
#     plot_matrix(B, "B Matrix (Masked)", os.path.join(output_dir, "B_matrix.png"))
#     plot_matrix(C, "C Matrix (Masked)", os.path.join(output_dir, "C_matrix.png"))
########################################################

# Models/visualize_matrices.py after moving ymodule to input 
import torch
import matplotlib.pyplot as plt
import os
import seaborn as sns
import logging # Added logging

def plot_matrix(matrix_data, title, save_path, aspect_auto=False):
    plt.figure(figsize=(10, 8))
    if aspect_auto:
        # For matrices like projection weights where aspect ratio is not square
        height, width = matrix_data.shape
        fig_width = 10
        fig_height = max(8, fig_width * height / width if width > 0 else fig_width) # Adjust height based on aspect ratio
        plt.figure(figsize=(fig_width, fig_height))
        sns.heatmap(matrix_data, cmap="viridis", cbar=True, square=False) # square=False for non-square matrices
    else:
        sns.heatmap(matrix_data, cmap="viridis", cbar=True, square=True) # square=True for A,B,C
    plt.title(title)
    plt.tight_layout()
    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Saved plot: {save_path}")

def save_and_plot_matrices(model, output_dir="logs/matrices_visualization"): # Changed default output_dir for clarity
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Saving and plotting matrices to {output_dir}")

    if not hasattr(model, 's4_layer'):
        logging.error("Model does not have an 's4_layer' attribute. Cannot save/plot S4 matrices.")
        return

    s4 = model.s4_layer

    # --- Save and Plot A, B, C matrices (related to S4 hidden state) ---
    # These matrices are (hidden_dim, hidden_dim), e.g. (273, 273)
    if hasattr(s4, 'A') and hasattr(s4, 'A_mask'):
        A_unmasked = s4.A.detach().cpu()
        A_masked_np = (s4.A * s4.A_mask).detach().cpu().numpy()
        torch.save(A_unmasked, os.path.join(output_dir, "S4_A_unmasked.pt"))
        plot_matrix(A_masked_np, "S4 A Matrix (Masked)", os.path.join(output_dir, "S4_A_masked.png"))
    else:
        logging.warning("S4 layer missing A or A_mask.")

    if hasattr(s4, 'B') and hasattr(s4, 'B_mask'):
        B_unmasked = s4.B.detach().cpu()
        B_masked_np = (s4.B * s4.B_mask).detach().cpu().numpy()
        torch.save(B_unmasked, os.path.join(output_dir, "S4_B_unmasked_internal.pt")) # B is internal to S4 state update
        plot_matrix(B_masked_np, "S4 B Matrix (Masked, Internal)", os.path.join(output_dir, "S4_B_masked_internal.png"))
    else:
        logging.warning("S4 layer missing B or B_mask.")

    if hasattr(s4, 'C') and hasattr(s4, 'C_mask'):
        C_unmasked = s4.C.detach().cpu()
        C_masked_np = (s4.C * s4.C_mask).detach().cpu().numpy()
        torch.save(C_unmasked, os.path.join(output_dir, "S4_C_unmasked.pt"))
        plot_matrix(C_masked_np, "S4 C Matrix (Masked)", os.path.join(output_dir, "S4_C_masked.png"))
    else:
        logging.warning("S4 layer missing C or C_mask.")

    # --- Save and Plot Input Projection Layer (if exists) ---
    if hasattr(s4, 'input_projection'):
        input_proj_layer = s4.input_projection
        # Save state dict (weights and biases)
        torch.save(input_proj_layer.state_dict(), os.path.join(output_dir, "S4_input_projection_state_dict.pt"))
        
        # Plot weights
        # Weight matrix shape: (s4.hidden_dim, s4.input_actual_dim), e.g. (273, 274)
        proj_weights_np = input_proj_layer.weight.detach().cpu().numpy()
        plot_matrix(proj_weights_np, f"S4 Input Projection Weights ({proj_weights_np.shape[0]}x{proj_weights_np.shape[1]})",
                    os.path.join(output_dir, "S4_input_projection_weights.png"), aspect_auto=True)
        
        if input_proj_layer.bias is not None:
            proj_bias_np = input_proj_layer.bias.detach().cpu().numpy()
            # Plotting bias as a 1D heatmap might not be very insightful, but possible
            # Or just save it. For now, let's just log its existence.
            logging.info(f"S4 Input Projection bias shape: {proj_bias_np.shape}")
            # torch.save(input_proj_layer.bias.detach().cpu(), os.path.join(output_dir, "S4_input_projection_bias.pt"))

    else:
        logging.warning("S4 layer does not have an 'input_projection' attribute.")
    
    logging.info("Finished saving and plotting matrices.")