import torch 
import numpy as np

def convert_file_to_dict(file_path: str) -> dict:
    """
    Convert a txt file to a dictionary. Should be used for converting the class map txt file to a dictionary.
    
    Args:
        file_path (str): The path to the txt file.
    
    Returns:
        dict: A dictionary with the class id as the key and the class name as the value.
    """
    result = {}
    try:
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 2:
                    path, number = parts
                    key = next(part for part in path.split('/') if part.startswith('n'))
                    if key not in result:
                        result[key] = int(number)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return result

def get_class_map(image_folder_path: str , 
                  val_map_path: str ) -> tuple:
    """
    Retrieve class mappings from the training dataset and validation map.

    Args:
        image_folder_path (str): Path to the training dataset folder.
        val_map_path (str): Path to the validation map file.

    Returns:
        tuple: A tuple containing two dictionaries:
            - folder_train_dict: Mapping of class names to indices from the training dataset.
            - zip_train_dict: Mapping of class names to indices from the validation map.
    """
    dataset = torchvision.datasets.ImageFolder(root=image_folder_path)
    folder_train_dict = dataset.class_to_idx
    zip_train_dict = convert_file_to_dict(val_map_path)
    
    return folder_train_dict, zip_train_dict

class PredictionConverter:
    def __init__(self, dict1: dict, dict2: dict):
        """
        Initialize the PredictionConverter with two dictionaries.
        
        Args:
            dict1 (dict): The first dictionary mapping keys to values.
            dict2 (dict): The second dictionary mapping keys to values.
        
        Raises:
            ValueError: If the keys of the two dictionaries are not identical.
        """
        if set(dict1.keys()) != set(dict2.keys()):
            raise ValueError("The keys of both dictionaries must be identical.")
        self.dict1 = dict1
        self.dict2 = dict2
        self.reverse_dict1 = {v: k for k, v in dict1.items()}
        self.reverse_dict2 = {v: k for k, v in dict2.items()}

    def convert(self, value: int) -> int:
        """Convert a value from dict2 to dict1."""
        if value == -1:
            return -1
        if value not in self.reverse_dict2:
            raise ValueError(f"Value '{value}' does not exist in dict2.")
        key = self.reverse_dict2[value]
        return self.dict1[key]
    
    def reverse_convert(self, value: int) -> int:
        """Convert a value from dict1 to dict2."""
        if value == -1:
            return -1
        if value not in self.reverse_dict1:
            raise ValueError(f"Value '{value}' does not exist in dict1.")
        key = self.reverse_dict1[value]
        return self.dict2[key]
    
    def batch_convert(self, values, reverse: bool = False):
        """
        Convert a batch of values.
        
        Args:
            values: Input values (torch.Tensor, np.ndarray, list, or int).
            reverse (bool): If True, convert from dict1 to dict2. Otherwise, dict2 to dict1.
        
        Returns:
            Converted values in the same format as input.
        """
        convert_func = self.reverse_convert if reverse else self.convert
        
        if isinstance(values, torch.Tensor):
            return self._convert_tensor(values, convert_func)
        elif isinstance(values, np.ndarray):
            return self._convert_numpy(values, convert_func)
        elif isinstance(values, list):
            return self._convert_list(values, convert_func)
        elif isinstance(values, int):
            return convert_func(values)
        else:
            raise ValueError("Input must be a torch.Tensor, np.ndarray, list, or int.")
    
    def _convert_tensor(self, tensor: torch.Tensor, convert_func) -> torch.Tensor:
        return torch.tensor([convert_func(int(v)) for v in tensor.flatten()]).reshape(tensor.shape)
    
    def _convert_numpy(self, array: np.ndarray, convert_func) -> np.ndarray:
        return np.array([convert_func(int(v)) for v in array.flatten()]).reshape(array.shape)
    
    def _convert_list(self, lst: list, convert_func) -> list:
        return [convert_func(v) for v in lst]


class ZipTrainingConverter(PredictionConverter):
    """
    A converter class that converts between a dictionary loaded from a file and a target dictionary.

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

    Note: 
    You can use any target_dict in this convert. For Torchvision with ImageNet dataset. Use 'folder_map, zip_map = 
    get_class_map(folder_dataset_path, 'zip_training_converter/zip_val_map.txt') to get proporate classmap.' 

    """
    def __init__(self, , target_dict: dict, zip_file_path = 'zip_training_converter/zip_val_map.txt'):
        """
        Initialize the ZipTrainingConverter with a file path and target dictionary.
        
        Args:
            file_path (str): Path to the txt file containing the source mapping.
            target_dict (dict): The target dictionary to convert to/from.
        """
        # Convert the file to dictionary
        source_dict = convert_file_to_dict(zip_file_path)
        
        # Check if the file was successfully converted
        if not source_dict:
            raise ValueError(f"Failed to load dictionary from file: {zip_file_path}")
            
        # Initialize the parent class with both dictionaries
        super().__init__(source_dict, target_dict)

