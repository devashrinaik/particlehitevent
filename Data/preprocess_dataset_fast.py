# import os, json, torch
# import numpy as np
# from tqdm import tqdm
# from multiprocessing import Pool, cpu_count

# DATA_DIR = "training_data"
# SAVE_DIR = "preprocessed_data"
# SEQUENCE_LENGTH = 80
# INPUT_DIM = 273
# TARGET_DIM = 9
# LOG_FILE = os.path.join(SAVE_DIR, "preprocess_debug.log")

# os.makedirs(SAVE_DIR, exist_ok=True)

# def log(msg):
#     with open(LOG_FILE, "a") as f:
#         f.write(msg + "\n")
#     print(msg)

# def parse_file(file_path):
#     file_inputs, file_targets = [], []
#     error_count = 0
#     try:
#         with open(file_path, "r") as f:
#             lines = f.readlines()

#         cluster_indices = [i for i, line in enumerate(lines) if line.strip() == "<cluster>"]

#         for cluster_idx in cluster_indices:
#             try:
#                 target_line = lines[cluster_idx + 1].strip()
#                 target = np.array(list(map(float, target_line.split())), dtype=np.float32)
#                 if len(target) != TARGET_DIM:
#                     raise ValueError(f"Target length {len(target)} != {TARGET_DIM}")

#                 frames = []
#                 # for i, line in enumerate(lines):
#                 #     if line.startswith("<time slice"):
#                 #         frame = [list(map(float, lines[i + j + 1].strip().split())) for j in range(13)]
#                 #         frames.append(np.array(frame).flatten())

#                 # Get end index of the current cluster (start of next cluster or end of file)
#                 end_idx = next((i for i in range(cluster_idx + 1, len(lines)) if lines[i].strip() == "<cluster>"), len(lines))

#                 frames = []
#                 i = cluster_idx + 2  # start just after the target line
#                 while i < end_idx:
#                     if lines[i].startswith("<time slice"):
#                         try:
#                             frame = [list(map(float, lines[i + j + 1].strip().split())) for j in range(13)]
#                             frames.append(np.array(frame).flatten())
#                             i += 14  # skip current slice + 13 rows
#                         except (IndexError, ValueError):
#                             break
#                     else:
#                         i += 1


#                 if len(frames) != SEQUENCE_LENGTH:
#                     raise ValueError(f"Expected 80 frames, got {len(frames)}")

#                 input_tensor = np.array(frames, dtype=np.float32)
#                 file_inputs.append(input_tensor)
#                 file_targets.append(target)

#             except Exception as e:
#                 error_count += 1
#                 log(f"[WARN] Skipped cluster in {os.path.basename(file_path)}: {e}")

#         return file_inputs, file_targets, os.path.basename(file_path), error_count

#     except Exception as e:
#         log(f"[ERROR] Failed to parse file {file_path}: {e}")
#         return [], [], os.path.basename(file_path), -1

# def main():
#     if os.path.exists(LOG_FILE):
#         os.remove(LOG_FILE)

#     files = sorted([
#         os.path.join(DATA_DIR, f)
#         for f in os.listdir(DATA_DIR)
#         if f.startswith("pixel_clusters") and f.endswith(".out")
#     ])

#     log(f"[INFO] Found {len(files)} files. Parsing with {cpu_count()} workers...")

#     all_inputs, all_targets = [], []
#     total_errors = 0
#     corrupted_files = []

#     with Pool(cpu_count()) as pool:
#         results = list(tqdm(pool.imap(parse_file, files), total=len(files)))

#     for inputs, targets, fname, err_count in results:
#         all_inputs.extend(inputs)
#         all_targets.extend(targets)
#         if err_count == -1:
#             corrupted_files.append(fname)
#         else:
#             total_errors += err_count

#     log(f"[INFO] Total clusters processed: {len(all_inputs)}")
#     log(f"[INFO] Total clusters skipped due to errors: {total_errors}")
#     log(f"[INFO] Corrupted or unreadable files: {corrupted_files}")

#     if not all_inputs or not all_targets:
#         log("[FATAL] No valid data extracted. Aborting.")
#         return

#     # Convert to CPU tensors for normalization
#     inputs_tensor = torch.tensor(np.stack(all_inputs), dtype=torch.float32)
#     targets_tensor = torch.tensor(np.stack(all_targets), dtype=torch.float32)

#     # Normalize globally
#     input_min, input_max = inputs_tensor.min(), inputs_tensor.max()
#     target_min, target_max = targets_tensor.min(0).values, targets_tensor.max(0).values

#     norm_inputs = (inputs_tensor - input_min) / (input_max - input_min + 1e-8)
#     norm_targets = (targets_tensor - target_min) / (target_max - target_min + 1e-8)

#     # Save stats for later inverse normalization
#     with open(os.path.join(SAVE_DIR, "norm_stats.json"), "w") as f:
#         json.dump({
#             "input_min": float(input_min),
#             "input_max": float(input_max),
#             "target_min": target_min.tolist(),
#             "target_max": target_max.tolist()
#         }, f)
#     log("[INFO] Saved normalization stats")

#     # Save dataset in chunks
#     chunk_size = 20000
#     os.makedirs(SAVE_DIR, exist_ok=True)
#     total_samples = norm_inputs.shape[0]

#     for i in range(0, total_samples, chunk_size):
#         chunk_inputs = norm_inputs[i:i + chunk_size]
#         chunk_targets = norm_targets[i:i + chunk_size]
#         torch.save(
#             {'inputs': chunk_inputs, 'targets': chunk_targets},
#             os.path.join(SAVE_DIR, f"chunk_{i // chunk_size:03}.pt")
#         )
#         log(f"[INFO] Saved chunk {i // chunk_size:03} with {len(chunk_inputs)} samples")

#     log(f"[SUCCESS] Preprocessing complete. All chunks saved in {SAVE_DIR}")


# if __name__ == "__main__":
#     main()
########################################################

# # Data/preprocess_dataset_fast.py before moving the ymodule to input 
# import os, json, torch
# import numpy as np
# from tqdm import tqdm
# from multiprocessing import Pool, cpu_count

# DATA_DIR = "training_data"
# SAVE_DIR = "preprocessed_data"
# SEQUENCE_LENGTH = 80
# INPUT_DIM = 273
# TARGET_DIM = 9
# CHUNK_SIZE = 20000
# LOG_FILE = os.path.join(SAVE_DIR, "preprocess_debug.log")

# os.makedirs(SAVE_DIR, exist_ok=True)

# def log(msg):
#     with open(LOG_FILE, "a") as f:
#         f.write(msg + "\n")
#     print(msg)

# def parse_file(file_path):
#     file_inputs, file_targets = [], []
#     error_count = 0
#     try:
#         with open(file_path, "r") as f:
#             lines = f.readlines()

#         cluster_indices = [i for i, line in enumerate(lines) if line.strip() == "<cluster>"]

#         for cluster_idx in cluster_indices:
#             try:
#                 target_line = lines[cluster_idx + 1].strip()
#                 target = np.array(list(map(float, target_line.split())), dtype=np.float32)
#                 if len(target) != TARGET_DIM:
#                     raise ValueError(f"Target length {len(target)} != {TARGET_DIM}")

#                 end_idx = next((i for i in range(cluster_idx + 1, len(lines)) if lines[i].strip() == "<cluster>"), len(lines))
#                 frames = []
#                 i = cluster_idx + 2
#                 while i < end_idx:
#                     if lines[i].startswith("<time slice"):
#                         try:
#                             frame = [list(map(float, lines[i + j + 1].strip().split())) for j in range(13)]
#                             frames.append(np.array(frame).flatten())
#                             i += 14
#                         except (IndexError, ValueError):
#                             break
#                     else:
#                         i += 1

#                 if len(frames) != SEQUENCE_LENGTH:
#                     raise ValueError(f"Expected {SEQUENCE_LENGTH} frames, got {len(frames)}")

#                 input_tensor = np.array(frames, dtype=np.float32)
#                 file_inputs.append(input_tensor)
#                 file_targets.append(target)

#             except Exception as e:
#                 error_count += 1
#                 log(f"[WARN] Skipped cluster in {os.path.basename(file_path)}: {e}")

#         return file_inputs, file_targets, os.path.basename(file_path), error_count

#     except Exception as e:
#         log(f"[ERROR] Failed to parse file {file_path}: {e}")
#         return [], [], os.path.basename(file_path), -1

# def compute_stats(results):
#     input_min = float("inf")
#     input_max = float("-inf")
#     target_min = np.full(TARGET_DIM, float("inf"))
#     target_max = np.full(TARGET_DIM, float("-inf"))

#     for inputs, targets, _, _ in results:
#         for x in inputs:
#             input_min = min(input_min, np.min(x))   # global across time/space
#             input_max = max(input_max, np.max(x))
#         for y in targets:
#             target_min = np.minimum(target_min, y)  # per-feature min
#             target_max = np.maximum(target_max, y)  # per-feature max

#     stats = {
#         "input_min": float(input_min),
#         "input_max": float(input_max),
#         "target_min": target_min.tolist(),
#         "target_max": target_max.tolist()
#     }

#     with open(os.path.join(SAVE_DIR, "norm_stats.json"), "w") as f:
#         json.dump(stats, f)

#     log("[INFO] Saved normalization stats")
#     return input_min, input_max, target_min, target_max


# def normalize_and_save_chunks(results, input_min, input_max, target_min, target_max):
#     buffer_inputs, buffer_targets = [], []
#     chunk_id = 0

#     for inputs, targets, _, _ in tqdm(results, desc="Normalizing and saving"):
#         for x, y in zip(inputs, targets):
#             # Global normalization for input
#             norm_x = (x - input_min) / (input_max - input_min + 1e-8)

#             # Per-feature normalization for target
#             norm_y = (y - target_min) / (target_max - target_min + 1e-8)

#             buffer_inputs.append(torch.tensor(norm_x, dtype=torch.float32))
#             buffer_targets.append(torch.tensor(norm_y, dtype=torch.float32))

#             if len(buffer_inputs) >= CHUNK_SIZE:
#                 torch.save({
#                     'inputs': torch.stack(buffer_inputs),
#                     'targets': torch.stack(buffer_targets)
#                 }, os.path.join(SAVE_DIR, f"chunk_{chunk_id:03}.pt"))
#                 log(f"[INFO] Saved chunk {chunk_id:03} with {len(buffer_inputs)} samples")
#                 buffer_inputs, buffer_targets = [], []
#                 chunk_id += 1

#     if buffer_inputs:
#         torch.save({
#             'inputs': torch.stack(buffer_inputs),
#             'targets': torch.stack(buffer_targets)
#         }, os.path.join(SAVE_DIR, f"chunk_{chunk_id:03}.pt"))
#         log(f"[INFO] Saved final chunk {chunk_id:03} with {len(buffer_inputs)} samples")

# def main():
#     if os.path.exists(LOG_FILE):
#         os.remove(LOG_FILE)

#     files = sorted([
#         os.path.join(DATA_DIR, f)
#         for f in os.listdir(DATA_DIR)
#         if f.startswith("pixel_clusters") and f.endswith(".out")
#     ])

#     log(f"[INFO] Found {len(files)} files. Parsing with {cpu_count()} workers...")

#     with Pool(cpu_count()) as pool:
#         results = list(tqdm(pool.imap(parse_file, files), total=len(files)))

#     total_clusters = sum(len(r[0]) for r in results)
#     total_errors = sum(r[3] for r in results if r[3] > 0)
#     corrupted_files = [r[2] for r in results if r[3] == -1]

#     log(f"[INFO] Total clusters processed: {total_clusters}")
#     log(f"[INFO] Total clusters skipped due to errors: {total_errors}")
#     log(f"[INFO] Corrupted or unreadable files: {corrupted_files}")

#     if total_clusters == 0:
#         log("[FATAL] No valid data extracted. Aborting.")
#         return

#     # Step 1: compute stats
#     input_min, input_max, target_min, target_max = compute_stats(results)

#     # Step 2: normalize and save
#     normalize_and_save_chunks(results, input_min, input_max, np.array(target_min), np.array(target_max))

#     log(f"[SUCCESS] Preprocessing complete. All chunks saved in {SAVE_DIR}")

# if __name__ == "__main__":
#     main()

########################################################

# # Data/preprocess_dataset_fast.py after moving the ymodule to input 
# import os
# import json
# import torch
# import numpy as np
# from tqdm import tqdm
# from multiprocessing import Pool, cpu_count

# # Configuration
# DATA_DIR = "training_data"
# SAVE_DIR = "preprocessed_data"
# SEQUENCE_LENGTH = 80
# ORIGINAL_TARGET_DIM = 9
# NEW_TARGET_DIM = ORIGINAL_TARGET_DIM - 1
# CHUNK_SIZE = 20000
# LOG_FILE = os.path.join(SAVE_DIR, "preprocess_debug.log")

# os.makedirs(SAVE_DIR, exist_ok=True)

# def log(msg):
#     with open(LOG_FILE, "a") as f:
#         f.write(msg + "\n")
#     print(msg)


# def parse_file(file_path):
#     file_inputs, file_targets = [], []
#     error_count = 0
#     try:
#         with open(file_path, "r") as f:
#             lines = f.readlines()

#         # Find clusters
#         cluster_indices = [i for i, line in enumerate(lines) if line.strip() == "<cluster>"]

#         for cluster_idx in cluster_indices:
#             try:
#                 # Original 9-dim target
#                 target_line = lines[cluster_idx + 1].strip()
#                 target = np.array(list(map(float, target_line.split())), dtype=np.float32)
#                 if len(target) != ORIGINAL_TARGET_DIM:
#                     raise ValueError(f"Target length {len(target)} != {ORIGINAL_TARGET_DIM}")

#                 # Collect 80 frames
#                 end_idx = next((i for i in range(cluster_idx + 1, len(lines))
#                                  if lines[i].strip() == "<cluster>"), len(lines))
#                 frames = []
#                 i = cluster_idx + 2
#                 while i < end_idx:
#                     if lines[i].startswith("<time slice"):
#                         # Read next 13 lines of pixel data
#                         frame = [list(map(float, lines[i + j + 1].split())) for j in range(13)]
#                         frames.append(np.array(frame).flatten())
#                         i += 14
#                     else:
#                         i += 1

#                 if len(frames) != SEQUENCE_LENGTH:
#                     raise ValueError(f"Expected {SEQUENCE_LENGTH} frames, got {len(frames)}")

#                 # Build input tensor: (80, 273)
#                 input_tensor = np.array(frames, dtype=np.float32)

#                 # Promote target index 7 into inputs as new feature per time-step
#                 moved_val = target[7]
#                 extra_col = np.full((len(frames), 1), moved_val, dtype=np.float32)
#                 input_tensor = np.concatenate([input_tensor, extra_col], axis=1)  # now (80, 274)

#                 # Remove promoted target from output
#                 target_remainder = np.concatenate([target[:7], target[8:]], axis=0)  # (8,)

#                 file_inputs.append(input_tensor)
#                 file_targets.append(target_remainder)

#             except Exception as e:
#                 error_count += 1
#                 log(f"[WARN] Skipped cluster in {os.path.basename(file_path)}: {e}")

#         return file_inputs, file_targets, os.path.basename(file_path), error_count

#     except Exception as e:
#         log(f"[ERROR] Failed to parse file {file_path}: {e}")
#         return [], [], os.path.basename(file_path), -1


# def compute_stats(results):
#     # Initialize global stats
#     input_min = float("inf")
#     input_max = float("-inf")
#     target_min = np.full(NEW_TARGET_DIM, float("inf"))
#     target_max = np.full(NEW_TARGET_DIM, float("-inf"))

#     # Accumulate
#     for inputs, targets, _, _ in results:
#         for x in inputs:
#             input_min = min(input_min, np.min(x))
#             input_max = max(input_max, np.max(x))
#         for y in targets:
#             target_min = np.minimum(target_min, y)
#             target_max = np.maximum(target_max, y)

#     # Save stats
#     stats = {
#         "input_min": float(input_min),
#         "input_max": float(input_max),
#         "target_min": target_min.tolist(),
#         "target_max": target_max.tolist(),
#         "input_dim": inputs[0].shape[1],
#         "target_dim": NEW_TARGET_DIM
#     }
#     with open(os.path.join(SAVE_DIR, "norm_stats.json"), "w") as f:
#         json.dump(stats, f)

#     log("[INFO] Saved normalization stats")
#     return input_min, input_max, target_min, target_max


# def normalize_and_save_chunks(results, input_min, input_max, target_min, target_max):
#     buffer_inputs, buffer_targets = [], []
#     chunk_id = 0

#     for inputs, targets, _, _ in tqdm(results, desc="Normalizing and saving"):
#         for x, y in zip(inputs, targets):
#             # Global normalization for inputs
#             norm_x = (x - input_min) / (input_max - input_min + 1e-8)
#             # Per-feature normalization for targets
#             norm_y = (y - target_min) / (target_max - target_min + 1e-8)

#             buffer_inputs.append(torch.tensor(norm_x, dtype=torch.float32))
#             buffer_targets.append(torch.tensor(norm_y, dtype=torch.float32))

#             if len(buffer_inputs) >= CHUNK_SIZE:
#                 torch.save({
#                     'inputs': torch.stack(buffer_inputs),
#                     'targets': torch.stack(buffer_targets)
#                 }, os.path.join(SAVE_DIR, f"chunk_{chunk_id:03}.pt"))
#                 log(f"[INFO] Saved chunk {chunk_id:03} with {len(buffer_inputs)} samples")
#                 buffer_inputs, buffer_targets = [], []
#                 chunk_id += 1

#     # Save any remaining
#     if buffer_inputs:
#         torch.save({
#             'inputs': torch.stack(buffer_inputs),
#             'targets': torch.stack(buffer_targets)
#         }, os.path.join(SAVE_DIR, f"chunk_{chunk_id:03}.pt"))
#         log(f"[INFO] Saved final chunk {chunk_id:03} with {len(buffer_inputs)} samples")


# def main():
#     if os.path.exists(LOG_FILE):
#         os.remove(LOG_FILE)

#     # Collect .out files
#     files = sorted([
#         os.path.join(DATA_DIR, f)
#         for f in os.listdir(DATA_DIR)
#         if f.startswith("pixel_clusters") and f.endswith(".out")
#     ])

#     log(f"[INFO] Found {len(files)} files. Parsing with {cpu_count()} workers...")

#     with Pool(cpu_count()) as pool:
#         results = list(tqdm(pool.imap(parse_file, files), total=len(files)))

#     total_clusters = sum(len(r[0]) for r in results)
#     total_errors = sum(r[3] for r in results if r[3] > 0)
#     corrupted_files = [r[2] for r in results if r[3] == -1]

#     log(f"[INFO] Total clusters processed: {total_clusters}")
#     log(f"[INFO] Total clusters skipped due to errors: {total_errors}")
#     log(f"[INFO] Corrupted or unreadable files: {corrupted_files}")

#     if total_clusters == 0:
#         log("[FATAL] No valid data extracted. Aborting.")
#         return

#     # Compute stats and normalize
#     input_min, input_max, target_min, target_max = compute_stats(results)
#     normalize_and_save_chunks(results, input_min, input_max, target_min, target_max)

#     log(f"[SUCCESS] Preprocessing complete. All chunks saved in {SAVE_DIR}")

# if __name__ == "__main__":
#     main()

# Data/preprocess_dataset_fast.py after moving the 8th target to input
import os, json, torch
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

DATA_DIR = "training_data"
SAVE_DIR = "preprocessed_data"
SEQUENCE_LENGTH = 80
INPUT_DIM_ORIGINAL = 273 # Original features per frame
TARGET_DIM_ORIGINAL = 9  # Original number of target values

# New dimensions
TARGET_DIM = TARGET_DIM_ORIGINAL - 1 # New target dimension (9 - 1 = 8)
INPUT_DIM = INPUT_DIM_ORIGINAL + 1   # New input dimension per frame (273 + 1 = 274)

CHUNK_SIZE = 20000
LOG_FILE = os.path.join(SAVE_DIR, "preprocess_debug.log")

os.makedirs(SAVE_DIR, exist_ok=True)

def log(msg):
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")
    print(msg)

def parse_file(file_path):
    file_inputs, file_targets_new, file_eighth_target_vals = [], [], []
    error_count = 0
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()

        cluster_indices = [i for i, line in enumerate(lines) if line.strip() == "<cluster>"]

        for cluster_idx in cluster_indices:
            try:
                target_line = lines[cluster_idx + 1].strip()
                target_full = np.array(list(map(float, target_line.split())), dtype=np.float32)
                if len(target_full) != TARGET_DIM_ORIGINAL:
                    raise ValueError(f"Original target length {len(target_full)} != {TARGET_DIM_ORIGINAL}")

                # Extract the 8th target value (index 7)
                eighth_target_value = target_full[7]
                # Create new target by removing the 8th value
                target_new = np.delete(target_full, 7)
                if len(target_new) != TARGET_DIM: # Should be 8
                    raise ValueError(f"New target length {len(target_new)} != {TARGET_DIM}")


                end_idx = next((i for i in range(cluster_idx + 1, len(lines)) if lines[i].strip() == "<cluster>"), len(lines))
                frames = []
                i = cluster_idx + 2
                while i < end_idx:
                    if lines[i].startswith("<time slice"):
                        try:
                            frame = [list(map(float, lines[i + j + 1].strip().split())) for j in range(13)]
                            frames.append(np.array(frame).flatten()) # Shape (INPUT_DIM_ORIGINAL,)
                            i += 14
                        except (IndexError, ValueError):
                            break
                    else:
                        i += 1

                if len(frames) != SEQUENCE_LENGTH:
                    raise ValueError(f"Expected {SEQUENCE_LENGTH} frames, got {len(frames)}")

                input_tensor_original = np.array(frames, dtype=np.float32) # Shape (SEQUENCE_LENGTH, INPUT_DIM_ORIGINAL)
                
                file_inputs.append(input_tensor_original)
                file_targets_new.append(target_new)
                file_eighth_target_vals.append(eighth_target_value)

            except Exception as e:
                error_count += 1
                log(f"[WARN] Skipped cluster in {os.path.basename(file_path)}: {e}")

        return file_inputs, file_targets_new, file_eighth_target_vals, os.path.basename(file_path), error_count

    except Exception as e:
        log(f"[ERROR] Failed to parse file {file_path}: {e}")
        return [], [], [], os.path.basename(file_path), -1

def compute_stats(results):
    input_min_val = float("inf")
    input_max_val = float("-inf")
    # Target stats are now for the reduced target vector
    target_min_arr = np.full(TARGET_DIM, float("inf"), dtype=np.float32)
    target_max_arr = np.full(TARGET_DIM, float("-inf"), dtype=np.float32)

    all_eighth_target_vals_as_input = []

    for inputs_original, targets_new, eighth_vals, _, _ in results:
        for x_orig in inputs_original:
            input_min_val = min(input_min_val, np.min(x_orig))
            input_max_val = max(input_max_val, np.max(x_orig))
        for y_new in targets_new:
            target_min_arr = np.minimum(target_min_arr, y_new)
            target_max_arr = np.maximum(target_max_arr, y_new)
        all_eighth_target_vals_as_input.extend(eighth_vals)

    # Include the eighth target value (now an input) in input stats calculation
    if all_eighth_target_vals_as_input:
        input_min_val = min(input_min_val, np.min(np.array(all_eighth_target_vals_as_input)))
        input_max_val = max(input_max_val, np.max(np.array(all_eighth_target_vals_as_input)))

    stats = {
        "input_min": float(input_min_val),    # Global scalar min for all input features (original + new one)
        "input_max": float(input_max_val),    # Global scalar max for all input features
        "target_min": target_min_arr.tolist(), # Per-feature for the new 8-dim target
        "target_max": target_max_arr.tolist()  # Per-feature for the new 8-dim target
    }

    with open(os.path.join(SAVE_DIR, "norm_stats.json"), "w") as f:
        json.dump(stats, f)

    log("[INFO] Saved normalization stats")
    return input_min_val, input_max_val, target_min_arr, target_max_arr


def normalize_and_save_chunks(results, input_min_stat, input_max_stat, target_min_stat, target_max_stat):
    buffer_inputs_final, buffer_targets_final = [], []
    chunk_id = 0

    # Convert target stats to numpy arrays for broadcasting
    target_min_np = np.array(target_min_stat, dtype=np.float32)
    target_max_np = np.array(target_max_stat, dtype=np.float32)

    for inputs_original_list, targets_new_list, eighth_target_vals_list, _, _ in tqdm(results, desc="Normalizing and saving"):
        for x_original, y_new, eighth_target_val_scalar in zip(inputs_original_list, targets_new_list, eighth_target_vals_list):
            # x_original shape: (SEQUENCE_LENGTH, INPUT_DIM_ORIGINAL) i.e. (80, 273)
            # y_new shape: (TARGET_DIM,) i.e. (8,)
            # eighth_target_val_scalar: scalar

            # Normalize original input features
            norm_x_original = (x_original - input_min_stat) / (input_max_stat - input_min_stat + 1e-8)

            # Normalize the eighth_target_val_scalar (which is now an input feature)
            # using the same global input statistics
            norm_eighth_target_val_scalar = (eighth_target_val_scalar - input_min_stat) / (input_max_stat - input_min_stat + 1e-8)
            
            # Expand this normalized scalar to match sequence length: (SEQUENCE_LENGTH, 1)
            new_input_feature_col = np.full((SEQUENCE_LENGTH, 1), norm_eighth_target_val_scalar, dtype=np.float32)

            # Concatenate normalized original inputs with the new normalized feature column
            # final_norm_x shape: (SEQUENCE_LENGTH, INPUT_DIM_ORIGINAL + 1) i.e. (80, 274)
            final_norm_x = np.concatenate((norm_x_original, new_input_feature_col), axis=1)

            # Normalize the new target vector (per-feature normalization)
            norm_y_new = (y_new - target_min_np) / (target_max_np - target_min_np + 1e-8)

            buffer_inputs_final.append(torch.tensor(final_norm_x, dtype=torch.float32))
            buffer_targets_final.append(torch.tensor(norm_y_new, dtype=torch.float32))

            if len(buffer_inputs_final) >= CHUNK_SIZE:
                torch.save({
                    'inputs': torch.stack(buffer_inputs_final), # inputs will have shape (CHUNK_SIZE, SEQ_LEN, INPUT_DIM)
                    'targets': torch.stack(buffer_targets_final) # targets will have shape (CHUNK_SIZE, TARGET_DIM)
                }, os.path.join(SAVE_DIR, f"chunk_{chunk_id:03}.pt"))
                log(f"[INFO] Saved chunk {chunk_id:03} with {len(buffer_inputs_final)} samples. Input shape per sample: {final_norm_x.shape}, Target shape per sample: {norm_y_new.shape}")
                buffer_inputs_final, buffer_targets_final = [], []
                chunk_id += 1

    if buffer_inputs_final:
        torch.save({
            'inputs': torch.stack(buffer_inputs_final),
            'targets': torch.stack(buffer_targets_final)
        }, os.path.join(SAVE_DIR, f"chunk_{chunk_id:03}.pt"))
        log(f"[INFO] Saved final chunk {chunk_id:03} with {len(buffer_inputs_final)} samples.")

def main():
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

    files = sorted([
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if f.startswith("pixel_clusters") and f.endswith(".out")
    ])

    log(f"[INFO] Found {len(files)} files. Parsing with {cpu_count()} workers...")
    log(f"[INFO] New INPUT_DIM per frame: {INPUT_DIM}, New TARGET_DIM: {TARGET_DIM}")


    with Pool(cpu_count()) as pool:
        # results will be list of (file_inputs, file_targets_new, file_eighth_target_vals, filename, error_count)
        results = list(tqdm(pool.imap(parse_file, files), total=len(files)))

    total_clusters = sum(len(r[0]) for r in results) # r[0] is file_inputs
    total_errors = sum(r[4] for r in results if r[4] > 0) # r[4] is error_count
    corrupted_files = [r[3] for r in results if r[4] == -1] # r[3] is filename

    log(f"[INFO] Total clusters processed: {total_clusters}")
    log(f"[INFO] Total clusters skipped due to errors: {total_errors}")
    if corrupted_files:
        log(f"[INFO] Corrupted or unreadable files: {corrupted_files}")

    if total_clusters == 0:
        log("[FATAL] No valid data extracted. Aborting.")
        return

    # Step 1: compute stats
    # target_min_stat, target_max_stat will be for the new TARGET_DIM
    input_min_s, input_max_s, target_min_s, target_max_s = compute_stats(results)
    log(f"[DEBUG] Computed Stats: input_min={input_min_s}, input_max={input_max_s}, target_min={target_min_s.tolist()}, target_max={target_max_s.tolist()}")


    # Step 2: normalize and save
    # Pass numpy arrays for target_min/max for easier broadcasting in normalization
    normalize_and_save_chunks(results, input_min_s, input_max_s, target_min_s, target_max_s)

    log(f"[SUCCESS] Preprocessing complete. All chunks saved in {SAVE_DIR}")

if __name__ == "__main__":
    main()
