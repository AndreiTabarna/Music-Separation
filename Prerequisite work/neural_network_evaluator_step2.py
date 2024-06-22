import json
import numpy as np
import glob

# Function to calculate median error for each instrument and error metric
def calculate_median_errors(json_files):
    # Dictionary to store median errors
    median_errors = {}

    # Iterate over each instrument
    for instrument in ['vocals', 'drums', 'bass', 'other']:
        # Dictionary to store errors for the current instrument
        instrument_errors = {}

        # Iterate over each error metric
        for metric in ['SI-SDR', 'SI-SIR', 'SI-SAR', 'SD-SDR', 'SNR', 'SRR',
                       'SI-SDRi', 'SD-SDRi', 'SNRi', 'MIX-SI-SDR', 'MIX-SD-SDR', 'MIX-SNR']:
            # List to store errors for the current metric
            metric_errors = []

            # Iterate over each JSON file
            for file in json_files:
                with open(file) as f:
                    data = json.load(f)
                    # Get the value for the metric
                    value = data[instrument][metric]
                    # If the value is not already a list, convert it to a list
                    if not isinstance(value, list):
                        value = [value]
                    # Append error for the current instrument and metric
                    metric_errors.extend(map(float, value))

            # Calculate median error for the current metric
            median_error = np.median(metric_errors)
            # Add median error to the dictionary
            instrument_errors[metric] = median_error

            # If overall category does not exist, create it
            if "overall" not in median_errors:
                median_errors["overall"] = {}
            
            # If the metric key does not exist in the overall category, create it
            if metric not in median_errors["overall"]:
                median_errors["overall"][metric] = 0
            
            # Add the median error to the overall score for the current metric
            median_errors["overall"][metric] += median_error

        # Add errors for the current instrument to the main dictionary
        median_errors[instrument] = instrument_errors

    return median_errors

# Get list of JSON files
json_files = glob.glob("*.json")

# Calculate median errors
median_errors = calculate_median_errors(json_files)

# Save results to a JSON file
with open("median_errors.json", "w") as f:
    json.dump(median_errors, f, indent=4)

print("Median errors saved to median_errors.json")

