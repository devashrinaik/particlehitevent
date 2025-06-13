# import os
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import mean_absolute_error, r2_score
# from Data.dataset import PixelClusterDataset
# from Models.s4network import S4PredictionNetwork


# def quantize_matrices(A, B, C, precision):
#     max_val = max(np.max(A), np.max(B), np.max(C))
#     min_val = min(np.min(A), np.min(B), np.min(C))

#     def quantize(matrix, min_val, max_val, precision):
#         normalized_matrix = (matrix - min_val) / (max_val - min_val)
#         quantized_matrix = np.round(normalized_matrix * (2**precision - 1)) / (2**precision - 1)
#         denormalized_matrix = quantized_matrix * (max_val - min_val) + min_val
#         return denormalized_matrix

#     A_quantized = quantize(A, min_val, max_val, precision)
#     B_quantized = quantize(B, min_val, max_val, precision)
#     C_quantized = quantize(C, min_val, max_val, precision)
#     return A_quantized, B_quantized, C_quantized


# def evaluate_model(model, dataloader, device):
#     model.eval()
#     preds, targets = [], []
#     with torch.no_grad():
#         for x, y in dataloader:
#             x, y = x.to(device), y.to(device)
#             output = model(x)
#             preds.append(output.cpu().numpy())
#             targets.append(y.cpu().numpy())

#     preds = np.concatenate(preds)
#     targets = np.concatenate(targets)
#     mae = mean_absolute_error(targets, preds, multioutput='uniform_average')
#     r2 = r2_score(targets, preds, multioutput='uniform_average')
#     return mae, r2, preds, targets


# def plot_predictions_comparison(original_preds, quantized_preds, targets, save_path="logs/quantized_vs_original.png"):
#     num_targets = targets.shape[1]
#     plt.figure(figsize=(6 * num_targets, 4))
#     for i in range(num_targets):
#         plt.subplot(1, num_targets, i + 1)
#         plt.plot(targets[:100, i], label="True", alpha=0.6)
#         plt.plot(original_preds[:100, i], label="Original", linestyle='--')
#         plt.plot(quantized_preds[:100, i], label="Quantized", linestyle=':')
#         plt.title(f"Target {i}")
#         plt.legend()
#     plt.tight_layout()
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     plt.savefig(save_path)
#     plt.close()


# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     data_dir = "preprocessed_data"
#     model = S4PredictionNetwork(input_dim=274, hidden_output_dim=128, final_output_dim=8).to(device)
#     model.load_state_dict(torch.load("logs/final_model_state_dict.pt", map_location=device))

#     dataset = PixelClusterDataset(data_dir, device=device)
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

#     original_mae, original_r2, original_preds, targets = evaluate_model(model, dataloader, device)

#     A = torch.load("logs/matrices_visualization/S4_A_unmasked.pt").numpy()
#     B = torch.load("logs/matrices_visualization/S4_B_unmasked_internal.pt").numpy()
#     C = torch.load("logs/matrices_visualization/S4_C_unmasked.pt").numpy()

#     A_q, B_q, C_q = quantize_matrices(A, B, C, precision=8)
#     model.s4_layer.A.data = torch.tensor(A_q, dtype=torch.float32).to(device)
#     model.s4_layer.B.data = torch.tensor(B_q, dtype=torch.float32).to(device)
#     model.s4_layer.C.data = torch.tensor(C_q, dtype=torch.float32).to(device)

#     quant_mae, quant_r2, quant_preds, _ = evaluate_model(model, dataloader, device)

#     print(f"Original MAE: {original_mae:.4f}, R2: {original_r2:.4f}")
#     print(f"Quantized MAE: {quant_mae:.4f}, R2: {quant_r2:.4f}")

#     plot_predictions_comparison(original_preds, quant_preds, targets)


# if __name__ == "__main__":
#     main()


# import os
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import mean_absolute_error, r2_score
# from Data.dataset import PixelClusterDataset
# from Models.s4network import S4PredictionNetwork

# def quantize_matrices_simulated(A, B, C, precision):
#     max_val = max(np.max(A), np.max(B), np.max(C))
#     min_val = min(np.min(A), np.min(B), np.min(C))

#     def quantize(matrix, min_val, max_val, precision):
#         normalized_matrix = (matrix - min_val) / (max_val - min_val)
#         quantized_matrix = np.round(normalized_matrix * (2**precision - 1)) / (2**precision - 1)
#         denormalized_matrix = quantized_matrix * (max_val - min_val) + min_val
#         return denormalized_matrix

#     A_quantized = quantize(A, min_val, max_val, precision)
#     B_quantized = quantize(B, min_val, max_val, precision)
#     C_quantized = quantize(C, min_val, max_val, precision)
#     return A_quantized, B_quantized, C_quantized

# def quantize_matrices_true_int8(A, B, C):
#     A_q = A.astype(np.float32).copy()
#     B_q = B.astype(np.float32).copy()
#     C_q = C.astype(np.float32).copy()

#     A_int8 = (A_q * 127).astype(np.int8)
#     B_int8 = (B_q * 127).astype(np.int8)
#     C_int8 = (C_q * 127).astype(np.int8)

#     A_restored = A_int8.astype(np.float32) / 127
#     B_restored = B_int8.astype(np.float32) / 127
#     C_restored = C_int8.astype(np.float32) / 127

#     return A_restored, B_restored, C_restored

# def evaluate_model(model, dataloader, device):
#     model.eval()
#     preds, targets = [], []
#     with torch.no_grad():
#         for x, y in dataloader:
#             x, y = x.to(device), y.to(device)
#             output = model(x)
#             preds.append(output.cpu().numpy())
#             targets.append(y.cpu().numpy())

#     preds = np.concatenate(preds)
#     targets = np.concatenate(targets)
#     mae = mean_absolute_error(targets, preds, multioutput='uniform_average')
#     r2 = r2_score(targets, preds, multioutput='uniform_average')
#     mae_per_target = mean_absolute_error(targets, preds, multioutput='raw_values')
#     r2_per_target = r2_score(targets, preds, multioutput='raw_values')
#     return mae, r2, preds, targets, mae_per_target, r2_per_target

# def plot_predictions_comparison(original_preds, simulated_preds, int8_preds, targets, save_path="logs/quantized_comparison.png"):
#     num_targets = targets.shape[1]
#     plt.figure(figsize=(6 * num_targets, 4))
#     for i in range(num_targets):
#         plt.subplot(1, num_targets, i + 1)
#         plt.plot(targets[:100, i], label="True", alpha=0.6)
#         plt.plot(original_preds[:100, i], label="Original", linestyle='--')
#         plt.plot(simulated_preds[:100, i], label="Simulated Quant", linestyle=':')
#         plt.plot(int8_preds[:100, i], label="True int8", linestyle='-.')
#         plt.title(f"Target {i}")
#         plt.legend()
#     plt.tight_layout()
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     plt.savefig(save_path)
#     plt.close()

# def save_matrices(A, B, C, directory):
#     os.makedirs(directory, exist_ok=True)
#     torch.save(torch.tensor(A), os.path.join(directory, "S4_A_quantized.pt"))
#     torch.save(torch.tensor(B), os.path.join(directory, "S4_B_quantized.pt"))
#     torch.save(torch.tensor(C), os.path.join(directory, "S4_C_quantized.pt"))

# def print_per_target_scores(mae_array, r2_array, label):
#     print(f"\n{label} Per-Target MAE and R2 Scores:")
#     for i, (mae, r2) in enumerate(zip(mae_array, r2_array)):
#         print(f"Target {i}: MAE={mae:.4f}, R2={r2:.4f}")

# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     data_dir = "preprocessed_data"
#     model = S4PredictionNetwork(input_dim=274, hidden_output_dim=128, final_output_dim=8).to(device)
#     model.load_state_dict(torch.load("logs/final_model_state_dict.pt", map_location=device))

#     dataset = PixelClusterDataset(data_dir, device=device)
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

#     original_mae, original_r2, original_preds, targets, orig_mae_pt, orig_r2_pt = evaluate_model(model, dataloader, device)

#     A = torch.load("logs/matrices_visualization/S4_A_unmasked.pt").numpy()
#     B = torch.load("logs/matrices_visualization/S4_B_unmasked_internal.pt").numpy()
#     C = torch.load("logs/matrices_visualization/S4_C_unmasked.pt").numpy()

#     # Simulated Quantization
#     A_q_sim, B_q_sim, C_q_sim = quantize_matrices_simulated(A, B, C, precision=8)
#     save_matrices(A_q_sim, B_q_sim, C_q_sim, "logs/quantized_matrices/simulated")
#     model.s4_layer.A.data = torch.tensor(A_q_sim, dtype=torch.float32).to(device)
#     model.s4_layer.B.data = torch.tensor(B_q_sim, dtype=torch.float32).to(device)
#     model.s4_layer.C.data = torch.tensor(C_q_sim, dtype=torch.float32).to(device)
#     sim_mae, sim_r2, sim_preds, _, sim_mae_pt, sim_r2_pt = evaluate_model(model, dataloader, device)

#     # True int8 Quantization
#     A_q_int8, B_q_int8, C_q_int8 = quantize_matrices_true_int8(A, B, C)
#     save_matrices(A_q_int8, B_q_int8, C_q_int8, "logs/quantized_matrices/true_int8")
#     model.s4_layer.A.data = torch.tensor(A_q_int8, dtype=torch.float32).to(device)
#     model.s4_layer.B.data = torch.tensor(B_q_int8, dtype=torch.float32).to(device)
#     model.s4_layer.C.data = torch.tensor(C_q_int8, dtype=torch.float32).to(device)
#     int8_mae, int8_r2, int8_preds, _, int8_mae_pt, int8_r2_pt = evaluate_model(model, dataloader, device)

#     print(f"\nOriginal MAE: {original_mae:.4f}, R2: {original_r2:.4f}")
#     print(f"Simulated Quant MAE: {sim_mae:.4f}, R2: {sim_r2:.4f}")
#     print(f"True int8 Quant MAE: {int8_mae:.4f}, R2: {int8_r2:.4f}")

#     print_per_target_scores(orig_mae_pt, orig_r2_pt, "Original")
#     print_per_target_scores(sim_mae_pt, sim_r2_pt, "Simulated Quant")
#     print_per_target_scores(int8_mae_pt, int8_r2_pt, "True int8 Quant")

#     plot_predictions_comparison(original_preds, sim_preds, int8_preds, targets)

# if __name__ == "__main__":
#     main()


# import os
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import mean_absolute_error, r2_score
# from Data.dataset import PixelClusterDataset
# from Models.s4network import S4PredictionNetwork

# def quantize_matrices_simulated(A, B, C, precision):
#     max_val = max(np.max(A), np.max(B), np.max(C))
#     min_val = min(np.min(A), np.min(B), np.min(C))

#     def quantize(matrix, min_val, max_val, precision):
#         normalized_matrix = (matrix - min_val) / (max_val - min_val)
#         quantized_matrix = np.round(normalized_matrix * (2**precision - 1)) / (2**precision - 1)
#         denormalized_matrix = quantized_matrix * (max_val - min_val) + min_val
#         return denormalized_matrix

#     A_quantized = quantize(A, min_val, max_val, precision)
#     B_quantized = quantize(B, min_val, max_val, precision)
#     C_quantized = quantize(C, min_val, max_val, precision)
#     return A_quantized, B_quantized, C_quantized

# def evaluate_model(model, dataloader, device):
#     model.eval()
#     preds, targets = [], []
#     with torch.no_grad():
#         for x, y in dataloader:
#             x, y = x.to(device), y.to(device)
#             output = model(x)
#             preds.append(output.cpu().numpy())
#             targets.append(y.cpu().numpy())

#     preds = np.concatenate(preds)
#     targets = np.concatenate(targets)
#     mae = mean_absolute_error(targets, preds, multioutput='uniform_average')
#     r2 = r2_score(targets, preds, multioutput='uniform_average')
#     mae_per_target = mean_absolute_error(targets, preds, multioutput='raw_values')
#     r2_per_target = r2_score(targets, preds, multioutput='raw_values')
#     return mae, r2, preds, targets, mae_per_target, r2_per_target

# def save_matrices(A, B, C, directory):
#     os.makedirs(directory, exist_ok=True)
#     torch.save(torch.tensor(A), os.path.join(directory, "S4_A_quantized.pt"))
#     torch.save(torch.tensor(B), os.path.join(directory, "S4_B_quantized.pt"))
#     torch.save(torch.tensor(C), os.path.join(directory, "S4_C_quantized.pt"))

# def print_per_target_scores(mae_array, r2_array, label):
#     print(f"\n{label} Per-Target MAE and R2 Scores:")
#     for i, (mae, r2) in enumerate(zip(mae_array, r2_array)):
#         print(f"Target {i}: MAE={mae:.4f}, R2={r2:.4f}")

# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     data_dir = "preprocessed_data"
#     model = S4PredictionNetwork(input_dim=274, hidden_output_dim=128, final_output_dim=8).to(device)
#     model.load_state_dict(torch.load("logs/final_model_state_dict.pt", map_location=device))

#     dataset = PixelClusterDataset(data_dir, device=device)
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

#     A = torch.load("logs/matrices_visualization/S4_A_unmasked.pt").numpy()
#     B = torch.load("logs/matrices_visualization/S4_B_unmasked_internal.pt").numpy()
#     C = torch.load("logs/matrices_visualization/S4_C_unmasked.pt").numpy()

#     original_mae, original_r2, original_preds, targets, orig_mae_pt, orig_r2_pt = evaluate_model(model, dataloader, device)
#     print(f"\nOriginal MAE: {original_mae:.4f}, R2: {original_r2:.4f}")
#     print_per_target_scores(orig_mae_pt, orig_r2_pt, "Original")

#     precisions = [2, 4, 6, 8]
#     for precision in precisions:
#         A_q, B_q, C_q = quantize_matrices_simulated(A, B, C, precision=precision)
#         quant_dir = f"logs/quantized_matrices/simulated_{precision}bit"
#         save_matrices(A_q, B_q, C_q, quant_dir)

#         model.s4_layer.A.data = torch.tensor(A_q, dtype=torch.float32).to(device)
#         model.s4_layer.B.data = torch.tensor(B_q, dtype=torch.float32).to(device)
#         model.s4_layer.C.data = torch.tensor(C_q, dtype=torch.float32).to(device)

#         mae, r2, preds, _, mae_pt, r2_pt = evaluate_model(model, dataloader, device)
#         print(f"\nSimulated {precision}-bit Quant MAE: {mae:.4f}, R2: {r2:.4f}")
#         print_per_target_scores(mae_pt, r2_pt, f"Simulated {precision}-bit Quant")

# if __name__ == "__main__":
#     main()


import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from Data.dataset import PixelClusterDataset
from Models.s4network import S4PredictionNetwork

def quantize_matrices_simulated(A, B, C, precision):
    max_val = max(np.max(A), np.max(B), np.max(C))
    min_val = min(np.min(A), np.min(B), np.min(C))

    def quantize(matrix, min_val, max_val, precision):
        normalized_matrix = (matrix - min_val) / (max_val - min_val)
        quantized_matrix = np.round(normalized_matrix * (2**precision - 1)) / (2**precision - 1)
        denormalized_matrix = quantized_matrix * (max_val - min_val) + min_val
        return denormalized_matrix

    A_quantized = quantize(A, min_val, max_val, precision)
    B_quantized = quantize(B, min_val, max_val, precision)
    C_quantized = quantize(C, min_val, max_val, precision)
    return A_quantized, B_quantized, C_quantized

def quantize_matrices_true_int(A, B, C, bits):
    scale = (2 ** (bits - 1)) - 1
    A_clipped = np.clip(A, -1, 1)
    B_clipped = np.clip(B, -1, 1)
    C_clipped = np.clip(C, -1, 1)
    A_q = (A_clipped * scale).astype(np.int8)
    B_q = (B_clipped * scale).astype(np.int8)
    C_q = (C_clipped * scale).astype(np.int8)
    return A_q, B_q, C_q

def evaluate_model(model, dataloader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            preds.append(output.cpu().numpy())
            targets.append(y.cpu().numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    mae = mean_absolute_error(targets, preds, multioutput='uniform_average')
    r2 = r2_score(targets, preds, multioutput='uniform_average')
    mae_per_target = mean_absolute_error(targets, preds, multioutput='raw_values')
    r2_per_target = r2_score(targets, preds, multioutput='raw_values')
    return mae, r2, preds, targets, mae_per_target, r2_per_target

def save_matrices(A, B, C, directory):
    os.makedirs(directory, exist_ok=True)
    torch.save(A, os.path.join(directory, "S4_A_quantized.pt"))
    torch.save(B, os.path.join(directory, "S4_B_quantized.pt"))
    torch.save(C, os.path.join(directory, "S4_C_quantized.pt"))

def print_per_target_scores(mae_array, r2_array, label):
    print(f"\n{label} Per-Target MAE and R2 Scores:")
    for i, (mae, r2) in enumerate(zip(mae_array, r2_array)):
        print(f"Target {i}: MAE={mae:.4f}, R2={r2:.4f}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "preprocessed_data"
    model = S4PredictionNetwork(input_dim=274, hidden_output_dim=128, final_output_dim=8).to(device)
    model.load_state_dict(torch.load("logs/final_model_state_dict.pt", map_location=device))

    dataset = PixelClusterDataset(data_dir, device=device)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

    original_mae, original_r2, original_preds, targets, orig_mae_pt, orig_r2_pt = evaluate_model(model, dataloader, device)

    A = torch.load("logs/matrices_visualization/S4_A_unmasked.pt").numpy()
    B = torch.load("logs/matrices_visualization/S4_B_unmasked_internal.pt").numpy()
    C = torch.load("logs/matrices_visualization/S4_C_unmasked.pt").numpy()

    for bits in [2, 4, 6, 8]:
        label = f"int{bits}"
        print(f"\nEvaluating True Quantization with {bits}-bit...")
        A_q, B_q, C_q = quantize_matrices_true_int(A, B, C, bits)
        save_matrices(torch.tensor(A_q, dtype=torch.int8), torch.tensor(B_q, dtype=torch.int8), torch.tensor(C_q, dtype=torch.int8), f"logs/quantized_matrices/{label}")

        model.s4_layer.A.data = torch.tensor(A_q, dtype=torch.int8).to(device).float()
        model.s4_layer.B.data = torch.tensor(B_q, dtype=torch.int8).to(device).float()
        model.s4_layer.C.data = torch.tensor(C_q, dtype=torch.int8).to(device).float()

        mae, r2, preds, _, mae_pt, r2_pt = evaluate_model(model, dataloader, device)
        print(f"{label} MAE: {mae:.4f}, R2: {r2:.4f}")
        print_per_target_scores(mae_pt, r2_pt, label)

if __name__ == "__main__":
    main()
