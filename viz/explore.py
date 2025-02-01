

import torch
from tqdm import tqdm
from collections import defaultdict
import numpy as np

def rewrite_forward(model, x: torch.Tensor, layer_name: str = 'avgpool') -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract features and logits from a ResNet-like model in a single forward pass.
    
    Args:
        model (torch.nn.Module): ResNet model from torchvision
        x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
        layer_name (str): Name of the layer to extract features from. Default is 'avgpool'
                         Options: ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 
                                 'layer3', 'layer4', 'avgpool']
    
    Returns:
        tuple[torch.Tensor, torch.Tensor]: (features, logits)
            - features: Tensor of shape (batch_size, feature_dim)
            - logits: Tensor of shape (batch_size, num_classes)
    """
    features = model.forward_features(x)
    try:
        logits = model.head_forward(features)
    except:
        logits = model.head(features)
    return features, logits


def calculate_accuracy_by_class(model, dataloader, num_classes: int, device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> list:
    """
    Calculate the accuracy of the model for each class in the dataset.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        dataloader (DataLoader): DataLoader containing the evaluation dataset.
        num_classes (int): Total number of classes in the dataset.
        device (str): Device to run the model on, defaults to 'cuda' if available.

    Returns:
        list: Sorted list of tuples containing class indices and their corresponding accuracy.
    """
    model.to(device)
    model.eval()  # Set model to evaluation mode

    # Initialize dictionaries to store correct predictions and total samples for each class
    correct_per_class = defaultdict(int)
    total_per_class = defaultdict(int)

    total_features = []
    total_labels = []

    # Disable gradient calculations
    with torch.no_grad():
        # Loop through the dataloader
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)

            # Get model predictions
            features, outputs = rewrite_forward(model, inputs)
            _, preds = torch.max(outputs, dim=-1)
            # Get current batch accuracy 
            correct = (preds == labels).sum().item()
            total = labels.size(0)  
            accuracy = correct / total
            tqdm.write(f"Accuracy: {accuracy:.4f}")

            total_features.append(features.cpu())
            total_labels.append(labels.cpu())

            # Update the correct and total counts for each class
            for label, pred in zip(labels, preds):
                total_per_class[label.item()] += 1
                correct_per_class[label.item()] += (label == pred).item()

    # Calculate accuracy for each class
    accuracy_per_class = {
        cls: correct_per_class[cls] / total_per_class[cls] if total_per_class[cls] > 0 else 0.0
        for cls in range(num_classes)
    }

    # Sort the accuracy by class from smallest to largest
    sorted_accuracy = sorted(accuracy_per_class.items(), key=lambda x: x[1])

    return sorted_accuracy, torch.cat(total_features, dim=0), torch.cat(total_labels, 0)

def list_differences_and_count_zeros(array_one, array_two):
    """
    Calculate differences between two arrays and count zero differences.

    Args:
        array_one (List[Tuple[int, float]]): First array of (class_id, accuracy) tuples.
        array_two (List[Tuple[int, float]]): Second array of (class_id, accuracy) tuples.

    Returns:
        Tuple[List[Tuple[int, float, float, float]], int]: 
            List of (class_id, difference, accuracy1, accuracy2) tuples and count of zero differences.
    """
    if len(array_one) != len(array_two):
        raise ValueError("The arrays must have the same length")

    # Convert to dictionaries for easier lookup
    dict_one = {item[0]: item[1] for item in array_one}
    dict_two = {item[0]: item[1] for item in array_two}
    
    # Check same keys 
    if dict_one.keys() != dict_two.keys():
        raise ValueError("The arrays must have the same keys after conversion to dictionaries")

    result = []
    positive_count = 0
    negative_count = 0
    zero_count = 0

    for class_id in sorted(dict_one.keys(), key=lambda x: int(x)): # They have same keys
        if class_id in dict_two:
            diff = dict_one[class_id] - dict_two[class_id]
            result.append((class_id, diff, dict_one[class_id], dict_two[class_id]))
            if abs(diff) < 1e-6:  # Use a small threshold for floating-point comparison
                zero_count += 1
            elif diff > 0:
                positive_count += 1
            else:
                negative_count += 1
        else:
            print(f"Warning: Class {class_id} not found in the second array")

    # Sort the result by class_id
    result.sort(key=lambda x: x[1], reverse=True)
    
    # Calculate average accuracy 
    accuracy_one = sum(dict_one.values()) / len(dict_one) * 100
    accuracy_two = sum(dict_two.values()) / len(dict_two) * 100

    return result, zero_count, positive_count, negative_count, accuracy_one, accuracy_two

import torch
from collections import Counter
from tqdm import tqdm

def predict_top_classes_with_error_analysis_from_model_predictions(model, dataloader, target_class, device='cuda'):
    """
    Predict the top 3 classes for images belonging to the target_class in the DataLoader.
    Calculate the number of incorrect predictions, error rate, and contributions of other classes during errors.
    
    Args:
    model: Trained model.
    dataloader: DataLoader containing image data.
    target_class: Class to analyze (int).
    num_classes: Total number of classes (int).
    device: Device to use, defaults to 'cuda' if available, otherwise 'cpu'.
    
    Returns:
    top3_frequent_classes: List of the three most frequent classes and their prediction counts (list of tuples).
    error_rate: Error rate of the model on the target_class (float).
    error_contributions: Contributions of the other two classes in the top 3 during errors (Counter object).
    top1_count: Count of top 1 predictions for each class (Counter object).
    """
    model.eval()  # Set model to evaluation mode
    model.to(device)
    
    top3_predictions = []
    total_target_class_images = 0
    incorrect_predictions = 0
    error_contributions = Counter()
    
    with torch.no_grad():  # Disable gradient calculations for faster inference
        for images, labels in tqdm(dataloader, desc="Predicting"):
            images, labels = images.to(device), labels.to(device)

            # Process only images belonging to the target_class
            mask = labels == target_class
            if mask.sum() == 0:
                continue  # Skip batch if no images belong to target_class
            
            filtered_images = images[mask]
            filtered_labels = labels[mask]
            total_target_class_images += filtered_labels.size(0)
            
            # Model predictions
            outputs = model(filtered_images)
            _, top3 = torch.topk(outputs, 3, dim=1)
            _, top1 = torch.max(outputs, 1)
            
            # Count incorrect predictions
            incorrect_mask = top1 != target_class
            incorrect_predictions += incorrect_mask.sum().item()
            
            # Record contributions of other classes during errors
            for i in range(filtered_images.size(0)):
                if incorrect_mask[i]:
                    error_contributions.update(top3[i][1:3].cpu().numpy().tolist())
            
            # Collect top3 predictions
            top3_predictions.extend(top3.cpu().numpy().tolist())
    
    # Flatten all top3 classes into a single list
    all_top3 = [cls for preds in top3_predictions for cls in preds]
    
    # Count occurrences of each class
    class_counter = Counter(all_top3)
    
    # Get the three most frequent classes and their counts
    top3_frequent_classes = class_counter.most_common(3)
    
    # Calculate error rate
    error_rate = incorrect_predictions / total_target_class_images if total_target_class_images > 0 else None
    
    return top3_frequent_classes, error_rate, error_contributions


def predict_top_classes_with_error_analysis_from_top1(model, dataloader, target_class, device='cuda'):
    """
    Predict the top 1 class for images belonging to the target_class in the DataLoader.
    Calculate the number of incorrect predictions and the error rate.
    
    Args:
    model: Trained model.
    dataloader: DataLoader containing image data.
    target_class: Class to analyze (int).
    num_classes: Total number of classes (int).
    device: Device to use, defaults to 'cuda' if available, otherwise 'cpu'.
    
    Returns:
    top1_frequent_classes: List of the most frequent class and its prediction count (list of tuples).
    error_rate: Error rate of the model on the target_class (float).
    """
    model.eval()  # Set model to evaluation mode
    model.to(device)
    
    top1_predictions = []
    total_target_class_images = 0
    incorrect_predictions = 0
    
    with torch.no_grad():  # Disable gradient calculations for faster inference
        for images, labels in tqdm(dataloader, desc="Predicting"):
            images, labels = images.to(device), labels.to(device)

            # Process only images belonging to the target_class
            mask = labels == target_class
            if mask.sum() == 0:
                continue  # Skip batch if no images belong to target_class
            
            filtered_images = images[mask]
            filtered_labels = labels[mask]
            total_target_class_images += filtered_labels.size(0)
            
            # Model predictions
            outputs = model(filtered_images)
            _, top1 = torch.max(outputs, 1)
            
            # Count incorrect predictions
            incorrect_mask = top1 != target_class
            incorrect_predictions += incorrect_mask.sum().item()
            
            # Collect top1 predictions
            top1_predictions.extend(top1.cpu().numpy().tolist())
    
    # Count occurrences of each class
    class_counter = Counter(top1_predictions)
    
    # Get the most frequent class and its count
    top1_frequent_classes = class_counter.most_common(1)
    
    # Calculate error rate
    error_rate = incorrect_predictions / total_target_class_images if total_target_class_images > 0 else None
    
    return top1_frequent_classes, error_rate

import torch
from tqdm import tqdm
from collections import Counter, defaultdict

def predict_top_classes_with_error_analysis(model, dataloader, num_classes, device='cuda'):
    """
    Predict the top classes for each class in the dataset from the DataLoader.
    Calculate the top 3 most frequent incorrect predictions for each class.
    
    Args:
    model: Trained model.
    dataloader: DataLoader containing image data.
    num_classes: Total number of classes (int).
    device: Device to use, defaults to 'cuda' if available, otherwise 'cpu'.
    
    Returns:
    top3_frequent_classes: Dictionary where keys are class indices and values are lists of tuples with 
                           the top 3 most frequent predicted classes and their counts.
    error_rates: Dictionary where keys are class indices and values are the error rates (floats).
    """
    model.eval()  # Set model to evaluation mode
    model.to(device)

    top1_predictions_by_class = defaultdict(list)
    total_images_by_class = defaultdict(int)
    incorrect_predictions_by_class = defaultdict(int)
    
    with torch.no_grad():  # Disable gradient calculations for faster inference
        for images, labels in tqdm(dataloader, desc="Predicting"):
            images, labels = images.to(device), labels.to(device)
            
            # Model predictions
            outputs = model(images)
            _, top1 = torch.max(outputs, 1)

            # Loop through each class and process the predictions
            for class_idx in range(num_classes):
                mask = labels == class_idx
                if mask.sum() == 0:
                    continue  # Skip if no images belong to this class in the batch
                
                filtered_images = images[mask]
                filtered_labels = labels[mask]
                total_images_by_class[class_idx] += filtered_labels.size(0)
                
                top1_for_class = top1[mask]
                
                # Count incorrect predictions
                incorrect_mask = top1_for_class != class_idx
                incorrect_predictions_by_class[class_idx] += incorrect_mask.sum().item()
                
                # Collect top1 predictions for this class
                top1_predictions_by_class[class_idx].extend(top1_for_class.cpu().numpy().tolist())
    
    # Calculate top 3 frequent incorrect classes for each target class
    top3_frequent_classes = {}
    error_rates = {}
    
    for class_idx in range(num_classes):
        class_counter = Counter(top1_predictions_by_class[class_idx])
        
        # Get the top 3 most frequent incorrect predictions
        top3_frequent_classes[class_idx] = class_counter.most_common(3)
        
        # Calculate error rate for the class
        total_class_images = total_images_by_class[class_idx]
        if total_class_images > 0:
            error_rate = incorrect_predictions_by_class[class_idx] / total_class_images
        else:
            error_rate = None
        
        error_rates[class_idx] = error_rate
    
    return top3_frequent_classes, error_rates

from collections import defaultdict

def find_confusion_pairs_and_groups(most3_tuples):
    """
    This function is used to detect confused pairs and strongly confused groups from most3 class relationships.
    
    Args:
    most3_tuples: List of tuples where each tuple is (class_index, [most_3 classes])
                  Example: [(0, [3, 4, 5]), (1, [2, 5, 3]), ...]
    
    Returns:
    confusion_pairs: List of tuples, where each tuple contains two class indices that form a confusion pair.
    strong_confusion_groups: List of sets, where each set contains three class indices that form a strong confusion group.
    """
    
    # Step 1: Build a reverse mapping for easier lookup of confusion relationships
    most3_dict = {class_idx: set(most3) for class_idx, most3 in most3_tuples}
    print(most3_dict)
    
    confusion_pairs = []
    strong_confusion_groups = []
    
    # Step 2: Find confusion pairs
    for class_a, most3_a in most3_dict.items():
        for class_b in most3_a:
            if class_b in most3_dict and class_a in most3_dict[class_b]:
                pair = tuple(sorted([class_a, class_b]))  # Ensure unique pairs
                if pair not in confusion_pairs:
                    confusion_pairs.append(pair)
    
    # Step 3: Find strong confusion groups (groups of 3 classes)
    # We loop through every possible pair of classes and see if there is a third class that forms a strong confusion group
    for i, (class_a, most3_a) in enumerate(most3_tuples):
        for j in range(i+1, len(most3_tuples)):
            class_b, most3_b = most3_tuples[j]
            
            # Check if class_a and class_b are already in a confusion pair
            if class_b in most3_a and class_a in most3_b:
                # Now check if there's a third class to form a strong confusion group
                common_classes = most3_a.intersection(most3_b)
                for class_c in common_classes:
                    if (class_c in most3_dict and 
                        class_a in most3_dict[class_c] and 
                        class_b in most3_dict[class_c]):
                        # If class_c is also confused with both class_a and class_b, we have a strong confusion group
                        group = {class_a, class_b, class_c}
                        if group not in strong_confusion_groups:
                            strong_confusion_groups.append(group)
    
    return confusion_pairs, strong_confusion_groups

def get_top3_differece_table(our_top3_frequent_classes, pretrained_top3_frequent_classes, our_accuracy_per_class, pretrained_accuracy_per_class, identify_classes, identify_type="smaller"):
    our_top3_frequent_classes_ = {key: value for key, value in our_top3_frequent_classes.items() if len(value) >= 3 and key in identify_classes}
    pretrained_top3_frequent_classes_ = {key: value for key, value in pretrained_top3_frequent_classes.items() if len(value) >= 3 and key in identify_classes}
    
    our_keys = set(our_top3_frequent_classes_.keys())
    pretrained_keys = set(pretrained_top3_frequent_classes_.keys())

    # Find keys unique to each set
    unique_to_our_keys = our_keys - pretrained_keys
    unique_to_pretrained_keys = pretrained_keys - our_keys
    common_keys = our_keys.intersection(pretrained_keys)

    print(f"Keys unique to our model: {unique_to_our_keys}")
    print(f"Keys unique to pretrained model: {unique_to_pretrained_keys}")
    print(f"Keys common to both models: {common_keys}")
    
    import prettytable as pt

    # Initialize PrettyTable with field names
    # Initialize PrettyTable with field names
    table = pt.PrettyTable()
    table.field_names = ["Class", "Our Top 3", "Pretrained Top 3", "Our Accuracy", "Pretrained Accuracy", "Difference"]
    
    comperable_keys = unique_to_our_keys if identify_type == "smaller" else unique_to_pretrained_keys
    
    # List to store rows for sorting
    rows = []
    
    # Populate the rows list with accuracy data
    for cls in comperable_keys:
        our_accuracy = [item[1] for item in our_accuracy_per_class if item[0] == cls]
        pretrained_accuracy = [item[1] for item in pretrained_accuracy_per_class if item[0] == cls]
        
        # Ensure accuracy lists are not empty before accessing
        if our_accuracy and pretrained_accuracy:
            difference = our_accuracy[0] - pretrained_accuracy[0]
            rows.append([
                cls,
                our_top3_frequent_classes[cls],
                pretrained_top3_frequent_classes[cls],
                our_accuracy[0],
                pretrained_accuracy[0],
                difference
            ])
    
    # Sort the rows by the absolute value of the difference
    rows.sort(key=lambda x: abs(x[5]), reverse=True)
    
    # Add sorted rows to the table
    for row in rows:
        table.add_row([row[0], row[1], row[2], row[3], row[4], f"{row[5]:.2f}"])
    
    # Print the table
    print(table)
    
def compare_model_predictions(dataloader, our_model, pretrained_model, focus_classes=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    our_model = our_model.to(device)
    pretrained_model = pretrained_model.to(device)
    
    our_model.eval()
    pretrained_model.eval()
    
    correct_ours_incorrect_pretrained = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Comparing predictions"):
            images, labels, paths = batch
            if focus_classes is not None:
                mask = labels == focus_classes
                if mask.sum() == 0:
                    continue
                images = images[mask]
                labels = labels[mask]
            images = images.to(device)
            labels = labels.to(device)
            
            # Get predictions from both models
            our_outputs = our_model(images)
            pretrained_outputs = pretrained_model(images)
            
            # Get the predicted classes
            our_preds = torch.argmax(our_outputs, dim=1)
            pretrained_preds = torch.argmax(pretrained_outputs, dim=1)
            
            # Compare predictions
            our_correct = our_preds == labels
            pretrained_correct = pretrained_preds == labels
            
            # Find indices where our model is correct and pretrained is incorrect
            indices = torch.where(our_correct & ~pretrained_correct)[0]
            
            for i in indices:
                correct_ours_incorrect_pretrained.append({
                    'path': paths[i],
                    'true_label': labels[i].item(),
                    'our_prediction': our_preds[i].item(),
                    'pretrained_prediction': pretrained_preds[i].item()
                })
    
    return correct_ours_incorrect_pretrained
