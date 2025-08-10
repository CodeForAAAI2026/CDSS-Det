import numpy as np
import sys

# Define the expected label mapping
expected_labels = {1, 2, 3, 4, 5, 0}  # Allowed labels after remapping

def verify_label_file(label_file):
    if not label_file.endswith(".npy"):
        print(f"Error: {label_file} is not a .npy file")
        return

    try:
        # Load label file
        label_data = np.load(label_file)
    except Exception as e:
        print(f"Error loading {label_file}: {e}")
        return

    # Find unique labels in the file
    unique_labels = set(np.unique(label_data))

    # Check if any unexpected labels exist
    unexpected_labels = unique_labels - expected_labels

    if unexpected_labels:
        print(f"Verification Failed: Unexpected labels found in {label_file}: {unexpected_labels}")
    else:
        print(f"Verification Passed: All labels in {label_file} are correctly updated.")

# Example usage:
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_label.py path/to/label.npy")
    else:
        verify_label_file(sys.argv[1])
