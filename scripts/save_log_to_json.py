import os
import json
import csv
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_scalar_data(log_dir):
    """
    Extract scalar data from a TensorBoard log directory and return it as a list of dictionaries.
    """
    event_accumulator = EventAccumulator(log_dir)
    event_accumulator.Reload()

    scalar_data = []
    tags = event_accumulator.Tags()["scalars"]

    for tag in tags:
        events = event_accumulator.Scalars(tag)
        for event in events:
            scalar_data.append({
                "tag": tag,
                "wall_time": event.wall_time,
                "step": event.step,
                "value": event.value
            })

    return scalar_data

def save_to_json(data, output_file):
    """
    Save the given data to a JSON file.
    """
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)

def save_to_csv(data, output_file):
    """
    Save the given data to a CSV file.
    """
    with open(output_file, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["tag", "wall_time", "step", "value"])
        writer.writeheader()
        for row in data:
            writer.writerow(row)

# Specify the log directory and output files
log_dir = "/dss/dsshome1/06/ge42vol2/OrganDETR/runs/ours_CT1k_full_msa"
json_output_file = "output.json"
csv_output_file = "output.csv"

# Extract scalar data from the log directory
scalar_data = extract_scalar_data(log_dir)

# Save the scalar data to JSON and CSV files
save_to_json(scalar_data, json_output_file)
save_to_csv(scalar_data, csv_output_file)

print(f"Data saved to {json_output_file} and {csv_output_file}")
