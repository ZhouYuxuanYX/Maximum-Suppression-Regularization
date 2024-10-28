This is a converter for different class mappting between folder training and zip file training.

```python 
Example:
# Create a target dictionary
target_dict = {'class1': 0, 'class2': 1, 'class3': 2}

# Initialize converter with file path and target dictionary
converter = ZipTrainingConverter('path/to/mapping.txt', target_dict)

# Convert from source to target
source_label = 'dog'
target_label = converter.convert_to_target(source_label)  # Returns corresponding target value

# Convert from target to source
target_value = 1
source_label = converter.convert_to_source(target_value)  # Returns corresponding source label
```

Note: 
You can use any target_dict in this convert. For Torchvision with ImageNet dataset. Use `folder_map, zip_map = 
get_class_map(folder_dataset_path, 'zip_training_converter/zip_val_map.txt') to get proporate classmap.`