import os
import pandas as pd

directory = "../datasets/translated/magpie_v2_llama3"

output_file = "merged.csv"

dataframes = []


for file in os.listdir(directory):
    if file.endswith(".csv"):
        file_path = os.path.join(directory, file)
        # Read each CSV file into a DataFrame
        df = pd.read_csv(file_path)
        dataframes.append(df)


merged_df = pd.concat(dataframes, ignore_index=True)

merged_file_path = os.path.join(directory, output_file)
merged_df.to_csv(merged_file_path, index=False)

for file in os.listdir(directory):
    if file.endswith(".csv") and file != output_file:
        os.remove(os.path.join(directory, file))

print(f"Merged CSV saved as {merged_file_path} and original files deleted.")