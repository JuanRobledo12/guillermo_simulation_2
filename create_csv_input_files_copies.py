import os
import shutil
import time

start_time = time.time()
# Paths
source_folder = "csv_input_files_for_jsons"  # Replace with your source folder containing the CSV files
csv_files = ["anc_corrected.csv", "demand_corrected.csv", "projects_final_v3.csv", "supply_corrected.csv"]  # Replace with your CSV file names

target_base_folder = "json_RDM"  # Replace with the directory containing the po{x} folders
# Iterate over each folder in the target directory
for folder_name in os.listdir(target_base_folder):
    folder_path = os.path.join(target_base_folder, folder_name)
    # Ensure it's a directory
    if os.path.isdir(folder_path):
        for csv_file in csv_files:
            source_file_path = os.path.join(source_folder, csv_file)
            destination_file_path = os.path.join(folder_path, csv_file)
            shutil.copy2(source_file_path, destination_file_path)  # Copies the file and preserves metadata
        print(f"Copied files to {folder_path}")

print("File copying completed.")
finish_time = time.time()
print(f"Total execution time: {finish_time - start_time} secs")