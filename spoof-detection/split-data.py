import os 
import random
import shutil 
from itertools import islice

output_directory = "dataset/split-data"
input_directory = "dataset/all"
classes = ["fake","real"]

# Remove existing directory if it exists
if os.path.exists(output_directory):
    shutil.rmtree(output_directory)
    print("Removed existing directory")

# Create necessary folders
folders = [
    "train/images", "train/labels",
    "val/images", "val/labels",
    "test/images", "test/labels"
]

for folder in folders:
    os.makedirs(os.path.join(output_directory, folder), exist_ok=True)

# Retrieve Unique File Names  
file_names = os.listdir(input_directory)

# Extract unique base names (without extensions)  
unique_names = list(set(name.split('.')[0] for name in file_names))
print(unique_names)
# Shuffle the List  
random.shuffle(unique_names)

# Compute Dataset Split Sizes 
split_ratio = {"train":0.7,"val":0.2,"test":0.1}
total_images = len(unique_names)
train_count = int(total_images * split_ratio['train'])
val_count = int(total_images * split_ratio['val'])
test_count = int(total_images * split_ratio['test'])

# Adjust for any rounding discrepancies  
remaining = total_images - (train_count + val_count + test_count)
train_count += remaining  

# Split Data into Groups  
split_sizes = [train_count, val_count, test_count]
iterator = iter(unique_names)
split_data = [list(islice(iterator, count)) for count in split_sizes]

print(f'Total Images: {total_images} \nSplit: {len(split_data[0])} {len(split_data[1])} {len(split_data[2])}')

# Copy Files to Respective Folders  
folders = ['train', 'val', 'test']
for i, subset in enumerate(split_data):
    for name in subset:
        shutil.copy(f'{input_directory}/{name}.jpg', f'{output_directory}/{folders[i]}/images/{name}.jpg')
        shutil.copy(f'{input_directory}/{name}.txt', f'{output_directory}/{folders[i]}/labels/{name}.txt')

print("Dataset split process completed successfully.")

# Create Data.yaml file 

data_yaml = f'path: ../Data\n\
train: ../train/images\n\
val: ../val/images\n\
test: ../test/images\n\
\n\
nc: {len(classes)}\n\
names: {classes}'


f = open(f"{output_directory}/data.yaml", 'a')
f.write(data_yaml)
f.close()

print("Data.yaml file Created...")