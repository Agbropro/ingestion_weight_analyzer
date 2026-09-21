import pandas as pd
import os
filename = "dataset_1.csv"
existing_data = 6000
added_data = 3000
total_data = existing_data + added_data

dataset_data = pd.read_csv(f"data/{filename}")

camera_count = len(dataset_data)

target_per_camera = total_data / camera_count

dataset_data["Deficit"] = (
    target_per_camera - dataset_data["Images"]
).clip(lower=0)

total_deficit = dataset_data["Deficit"].sum()

dataset_data["Weight"] = (
    dataset_data["Deficit"] / total_deficit
)

dataset_data["Weight_Percentage"] = (
    dataset_data["Weight"] * 100
)

# Raw allocation
dataset_data["Raw_Allocation"] = (
    dataset_data["Weight"] * added_data
)

# Start with floor allocation
dataset_data["Added_Images"] = (
    dataset_data["Raw_Allocation"]
    .astype(int)
)

# Number of images still unallocated
remaining = (
    added_data
    - dataset_data["Added_Images"].sum()
)

# Fractional remainder
dataset_data["Remainder"] = (
    dataset_data["Raw_Allocation"]
    - dataset_data["Added_Images"]
)

# Give remaining images to cameras with largest remainder
remainder_index = (
    dataset_data["Remainder"]
    .nlargest(remaining)
    .index
)

dataset_data.loc[
    remainder_index,
    "Added_Images",
] += 1

dataset_data["Final_Images"] = (
    dataset_data["Images"]
    + dataset_data["Added_Images"]
)

dataset_data["Current_Percentage"] = (
    dataset_data["Images"]
    / existing_data
    * 100
)

dataset_data["Final_Percentage"] = (
    dataset_data["Final_Images"]
    / total_data
    * 100
)

dataset_data = dataset_data.sort_values(
    "Weight",
    ascending=False,
)

result = dataset_data[
    [
        "NVR",
        "Camera",
        "Images",
        "Deficit",
        "Weight_Percentage",
        "Added_Images",
        "Final_Images",
        "Current_Percentage",
        "Final_Percentage",
    ]
]

print(f"Total camera      : {camera_count}")
print(f"Existing dataset  : {existing_data}")
print(f"New dataset       : {added_data}")
print(f"Final dataset     : {total_data}")
print(f"Target per camera : {target_per_camera:.2f}")
print(f"Allocated images  : {result['Added_Images'].sum()}")
print()

print(result.to_string(index=False))

os.makedirs("data/result",exist_ok=True)
result.to_csv(f"data/result/{filename}")
