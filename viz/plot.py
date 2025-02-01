import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
import matplotlib.patches as mpatches

def compute_class_projections(balanced_X, balanced_y, type='svd'):
    """
    Computes the projections of the penultimate layer activations onto a 2D plane formed by the class centroids.

    Parameters:
    - balanced_X (np.ndarray): Activation data.
    - balanced_y (np.ndarray): Class labels corresponding to the activations.

    Returns:
    - projections_chosen (np.ndarray): Selected projections for the chosen classes.
    - labels_chosen (np.ndarray): Corresponding labels for the selected projections.
    """
    if type == 'svd':
        # **Find Class Centroids**
        class_centroids = []
        focus_classes = np.unique(balanced_y)
        for cls in focus_classes:
            class_activations = balanced_X[balanced_y == cls]
            centroid = np.mean(class_activations, axis=0)
            class_centroids.append(centroid)
        class_centroids = np.array(class_centroids)

        # **Orthonormal Basis via SVD**
        u, s, vh = np.linalg.svd(class_centroids - np.mean(class_centroids, axis=0))
        orthonormal_basis = vh[:2]  # Top two vectors form the basis

        # **Project Activations onto 2D Plane**
        mean_activations = np.mean(balanced_X, axis=0)
        activations_centered = balanced_X - mean_activations
        projections = np.dot(activations_centered, orthonormal_basis.T)

        # **Select Projections for Chosen Classes**
        mask = np.isin(balanced_y, focus_classes)
        projections_chosen = projections[mask]
        labels_chosen = balanced_y[mask]

        return projections_chosen, labels_chosen
    elif type == 'tsne':
        tsne = TSNE(n_components=2, random_state=42, verbose=True)
        projections = tsne.fit_transform(balanced_X)
        return projections, balanced_y
    else:
        raise ValueError(f"Unsupported type: {type}")


def plot_class_projections(projectsions, labels, label_map=None,
                           x_lim=(-15, 15), y_lim=(-15, 15), figsize=(8, 6), post_fix='', save_path='/home/hengl/lbsm/vis/rebuttals/new_plot.pdf'):
    """
    Plots the projections of the penultimate layer activations onto a 2D plane.

    Parameters:
    - balanced_X (np.ndarray): Activation data.
    - balanced_y (np.ndarray): Class labels corresponding to the activations.
    - focus_classes (list): List of class labels to focus on.
    - label_map (dict, optional): Mapping from class labels to readable names.
    - x_lim (tuple, optional): Limits for the x-axis (default: (-15, 15)).
    - y_lim (tuple, optional): Limits for the y-axis (default: (-15, 15)).
    - figsize (tuple, optional): Size of the plot in inches (default: (8, 6)).
    """
    focus_classes = np.unique(labels)
    # **Map Labels to Readable Names**
    if label_map:
        labels_mapped = [label_map.get(int(l), f'Class {l}') for l in focus_classes]
    else:
        labels_mapped = [f'Class {l}' for l in focus_classes]

    # **Plotting**
    colors = ['red', 'green', 'blue']
    plt.figure(figsize=figsize)
    for i, cls in enumerate(focus_classes):
        class_mask = labels == cls
        plt.scatter(
            projectsions[class_mask, 0],
            projectsions[class_mask, 1],
            label=label_map.get(cls, f'Class {cls}') if label_map else f'Class {cls}',
            color=colors[i % len(colors)],
            alpha=0.6
        )
    
    plt.xlabel('Projection 1', fontsize=20)
    plt.ylabel('Projection 2', fontsize=20)
    plt.title(f'Projections {post_fix}', fontsize=22)
    plt.legend()
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    if save_path is not None:
        # Ensure the path ends with .pdf
        if not save_path.endswith('.pdf'):
            save_path = save_path + '.pdf'
        plt.savefig(save_path, bbox_inches='tight', format='pdf')
    plt.show()
    plt.close()

def extract_plot_ranges(X, pad=0.5, x_lim: tuple = (None, None), y_lim: tuple = (None, None)):
    """Extract plot ranges given 1D arrays of X and Y axis values, using specified limits if provided.
    
    This function checks if the provided x_lim and y_lim are within the min and max range of the data.
    
    Args:
        X (np.ndarray): Input data with shape (n_samples, 2).
        pad (float): Padding to add to the min and max values.
        x_lim (tuple): Limits for the x-axis (default: (None, None)).
        y_lim (tuple): Limits for the y-axis (default: (None, None)).
    
    Returns:
        tuple: (min_x1, max_x1, min_x2, max_x2) extracted plot ranges.
    """
    min_x1_data, max_x1_data = np.min(X[:, 0]), np.max(X[:, 0])
    min_x2_data, max_x2_data = np.min(X[:, 1]), np.max(X[:, 1])
    
    # if x_lim[0] is not None:
    #     if x_lim[0] < min_x1_data:
    #         print(f"Warning: x_lim[0] {x_lim[0]} is below the minimum {min_x1_data}. Limiting to {min_x1_data}.")
    #         x_lim = (min_x1_data, x_lim[1])
    #     elif x_lim[0] > max_x1_data:
    #         print(f"Warning: x_lim[0] {x_lim[0]} is above the maximum {max_x1_data}. Limiting to {max_x1_data}.")
    #         x_lim = (max_x1_data, x_lim[1])
    
    # if x_lim[1] is not None:
    #     if x_lim[1] < min_x1_data:
    #         print(f"Warning: x_lim[1] {x_lim[1]} is below the minimum {min_x1_data}. Limiting to {min_x1_data}.")
    #         x_lim = (x_lim[0], min_x1_data)
    #     elif x_lim[1] > max_x1_data:
    #         print(f"Warning: x_lim[1] {x_lim[1]} is above the maximum {max_x1_data}. Limiting to {max_x1_data}.")
    #         x_lim = (x_lim[0], max_x1_data)
    
    # if y_lim[0] is not None:
    #     if y_lim[0] < min_x2_data:
    #         print(f"Warning: y_lim[0] {y_lim[0]} is below the minimum {min_x2_data}. Limiting to {min_x2_data}.")
    #         y_lim = (min_x2_data, y_lim[1])
    #     elif y_lim[0] > max_x2_data:
    #         print(f"Warning: y_lim[0] {y_lim[0]} is above the maximum {max_x2_data}. Limiting to {max_x2_data}.")
    #         y_lim = (max_x2_data, y_lim[1])
    
    # if y_lim[1] is not None:
    #     if y_lim[1] < min_x2_data:
    #         print(f"Warning: y_lim[1] {y_lim[1]} is below the minimum {min_x2_data}. Limiting to {min_x2_data}.")
    #         y_lim = (y_lim[0], min_x2_data)
    #     elif y_lim[1] > max_x2_data:
    #         print(f"Warning: y_lim[1] {y_lim[1]} is above the maximum {max_x2_data}. Limiting to {max_x2_data}.")
    #         y_lim = (y_lim[0], max_x2_data)
    
    min_x1 = x_lim[0] if x_lim[0] is not None else min_x1_data - pad
    max_x1 = x_lim[1] if x_lim[1] is not None else max_x1_data + pad
    min_x2 = y_lim[0] if y_lim[0] is not None else min_x2_data - pad
    max_x2 = y_lim[1] if y_lim[1] is not None else max_x2_data + pad
    
    return min_x1, max_x1, min_x2, max_x2


def highlight_misclassified_samples(X_test_2d, y_test, y_pred, ax, class_color_mapping, size=100):
    """
    Highlight misclassified test samples on the plot.

    Args:
        X_test_2d (np.ndarray): 2D projection of test features.
        y_test (np.ndarray): True labels of test samples.
        y_pred (np.ndarray): Predicted labels of test samples.
        ax (matplotlib.axes.Axes): The axes object to plot on.
        class_color_mapping (dict): Mapping of class labels to colors.

    Returns:
        matplotlib.collections.PathCollection: Scatter plot of misclassified samples.
    """
    
    # if -1 in y_pred, create a mask to remove it
    if -1 in y_pred:
        mask = y_pred != -1
        y_pred = y_pred[mask]
        X_test_2d = X_test_2d[mask]
        y_test = y_test[mask]

    misclassified = y_test != y_pred
    misclassified_X = X_test_2d[misclassified]
    misclassified_y_true = y_test[misclassified]
    misclassified_y_pred = y_pred[misclassified]
    
    

    scatter_misclassified = ax.scatter(
        misclassified_X[:, 0], misclassified_X[:, 1],
        c=[class_color_mapping[label] for label in misclassified_y_pred],
        marker='X', s=size, edgecolor='k',
        label='Misclassified Test Data'
    )

    return scatter_misclassified

def generate_grid_points(min_x, max_x, min_y, max_y, resolution=100):
    """Generate resolution * resolution points within a given range."""
    xx, yy = np.meshgrid(np.linspace(min_x, max_x, resolution), 
                         np.linspace(min_y, max_y, resolution))
    return np.c_[xx.ravel(), yy.ravel()]


def downsample(X, sample_rate):
    """Downsample the data to a smaller size."""
    idx = np.random.choice(X.shape[0], size=int(X.shape[0] * sample_rate), replace=False)
    return idx


def plot_decision_boundary(classifier, features, features_2d, labels, test_size=0.2, random_state=42, post_fix='', 
                           highlight_miss=False, x_lim=(None, None), y_lim=(None, None), 
                           legend_position='outside', show_legend=True):
    """Plots the decision boundary of a classifier along with training and test data points.

    Args:
        classifier (sklearn.base.BaseEstimator): The classifier to evaluate.
        features (np.ndarray): Feature data for training and testing.
        features_2d (np.ndarray): 2D projection of the feature data for visualization.
        labels (np.ndarray): Class labels corresponding to the features.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.
        post_fix (str): Suffix for the plot title.
        highlight_miss (bool): Whether to highlight misclassified samples.
        x_lim (tuple): Limits for x-axis.
        y_lim (tuple): Limits for y-axis.
        legend_position (str): Position of the legend ('outside' or 'inside').
        show_legend (bool): Whether to show the legend.
    """
    
    cmap_background = "Pastel1"
    cmap_points = "Set1"
    marker_size_train = 60
    marker_size_test = 80
    alpha_background = 0.3
    alpha_train = 0.8
    alpha_test = 1.0
    padding = 0.5

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_state)
    X_train_2d, X_test_2d, _, _ = train_test_split(features_2d, labels, test_size=test_size, random_state=random_state)

    # Calculate accuracies
    train_accuracy = accuracy_score(y_train, classifier.predict(X_train))
    test_accuracy = accuracy_score(y_test, classifier.predict(X_test))
    assert train_accuracy == classifier.score(X_train, y_train) and test_accuracy == classifier.score(X_test, y_test), (
        f"Accuracy scores do not match: Train {train_accuracy:.4f}, Test {test_accuracy:.4f}"
    )

    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Create Voronoi tessellation for background
    voronoi_classifier = KNeighborsClassifier(n_neighbors=1, n_jobs=-1).fit(X_train_2d, y_train)
    y_test_predicted = classifier.predict(X_test)
    voronoi_test_accuracy = voronoi_classifier.score(X_test_2d, y_test_predicted)
    print(f"Voronoi test accuracy: {voronoi_test_accuracy:.2f}")

    # Generate grid points
    min_x1, max_x1, min_x2, max_x2 = extract_plot_ranges(X_train_2d, x_lim=x_lim, y_lim=y_lim)
    grid_points = generate_grid_points(min_x1, max_x1, min_x2, max_x2, resolution=125)
    print(f"Resolution: {int(np.sqrt(grid_points.shape[0]))}")

    # Predict background
    background_predictions = voronoi_classifier.predict(grid_points)

    # Unique classes and colors
    unique_classes = np.unique(labels)
    print(f"Unique classes: {unique_classes}")
    colors = plt.cm.get_cmap(cmap_points, len(unique_classes))
    class_color_mapping = {cls: colors(i) for i, cls in enumerate(unique_classes)}

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title(f"Decision Boundary {post_fix}", fontsize=14)

    # Plot background
    scatter_background = ax.scatter(
        grid_points[:, 0], grid_points[:, 1],
        c=[class_color_mapping[cls] for cls in background_predictions],
        cmap=cmap_background,
        alpha=alpha_background, s=30, label='Decision Boundary'
    )
    
    # Plot training points
    scatter_train = ax.scatter(
        X_train_2d[:, 0], X_train_2d[:, 1],
        c=[class_color_mapping[label] for label in y_train],
        edgecolor='k',
        s=marker_size_train, alpha=alpha_train, label='Training Data'
    )
    
    if not highlight_miss:
        # Plot test points
        scatter_test = ax.scatter(
            X_test_2d[:, 0], X_test_2d[:, 1],
            c=[class_color_mapping[label] for label in y_test],
            marker='X',
            edgecolor='k', s=marker_size_test, alpha=alpha_test, label='Test Data'
        )
    else:
        y_pred = classifier.predict(X_test)
        scatter_misclassified = highlight_misclassified_samples(X_test_2d, y_test, y_pred, ax, class_color_mapping)

    if show_legend:
        # Create legends
        category_handles = [mpatches.Patch(color=class_color_mapping[cls], label=f'Class {cls}') for cls in unique_classes]
        data_type_handles = [scatter_background, scatter_train]
        data_type_labels = ['Decision Boundary', 'Training Data']
        
        if not highlight_miss:
            data_type_handles.append(scatter_test)
            data_type_labels.append('Test Data')
        else:
            data_type_handles.append(scatter_misclassified)
            data_type_labels.append('Misclassified Test Data')

        if legend_position == 'outside':
            # Adjust the position of both legends outside the plot
            legend1 = ax.legend(handles=category_handles, title='Categories', loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=10)
            ax.add_artist(legend1)
            ax.legend(data_type_handles, data_type_labels, title='Data Types', loc='upper left', bbox_to_anchor=(1.01, 0.85), fontsize=10)
            plt.subplots_adjust(right=0.75)  # Adjust this value as needed
        else:
            # Place both legends inside the plot
            legend1 = ax.legend(handles=category_handles, title='Categories', loc='upper right', fontsize=10)
            ax.add_artist(legend1)
            ax.legend(data_type_handles, data_type_labels, title='Data Types', loc='lower right', fontsize=10)

    # Set axis labels and limits
    ax.set_xlabel('Projection 1', fontsize=12)
    ax.set_ylabel('Projection 2', fontsize=12)
    ax.set_xlim(min_x1-padding, max_x1+padding)
    ax.set_ylim(min_x2-padding, max_x2+padding)

    # Adjust the layout
    plt.tight_layout()
    plt.show()
    plt.close()


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def plot_decision_boundary1(classifier, features, features_2d, labels, test_size=0.2, random_state=42, post_fix='', 
                           highlight_miss=False, x_lim=(None, None), y_lim=(None, None), 
                           legend_position='outside', show_legend=True, other_classifier=None, other_features=None, show_title=False, save_path=None):
    """Plots the decision boundary of a classifier along with training and test data points."""
    
    cmap_background = "Pastel1"
    cmap_points = "Set1"
    marker_size_train = 80 * 40 / 60  # Increased from 60 to 80
    marker_size_test = 100 * 40 / 60  # Increased from 80 to 100
    alpha_background = 0.3
    alpha_train = 0.8
    alpha_test = 1.0
    padding = 0.5

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_state)
    X_train_2d, X_test_2d, _, _ = train_test_split(features_2d, labels, test_size=test_size, random_state=random_state)

    # Calculate accuracies
    train_accuracy = accuracy_score(y_train, classifier.predict(X_train))
    test_accuracy = accuracy_score(y_test, classifier.predict(X_test))
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Create Voronoi tessellation for background
    voronoi_classifier = KNeighborsClassifier(n_neighbors=1, n_jobs=-1).fit(X_train_2d, y_train)
    y_test_predicted = classifier.predict(X_test)
    voronoi_test_accuracy = voronoi_classifier.score(X_test_2d, y_test_predicted)
    print(f"Voronoi test accuracy: {voronoi_test_accuracy:.2f}")

    # Generate grid points
    min_x1, max_x1 = x_lim if None not in x_lim else (X_train_2d[:, 0].min(), X_train_2d[:, 0].max())
    min_x2, max_x2 = y_lim if None not in y_lim else (X_train_2d[:, 1].min(), X_train_2d[:, 1].max())
    xx, yy = np.meshgrid(np.linspace(min_x1, max_x1, 125), np.linspace(min_x2, max_x2, 125))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Predict background
    background_predictions = voronoi_classifier.predict(grid_points)

    # Unique classes and colors
    unique_classes = np.unique(labels)
    print(f"Unique classes: {unique_classes}")
    colors = ["red", "blue", "gray", "yellow", "purple", "orange", "pink", "brown", "gray", "olive"]
    class_color_mapping = {cls: colors[i] for i, cls in enumerate(unique_classes)}

    # Plotting
    plt.rcParams['font.weight'] = 'normal'  # Set global font weight to normal
    plt.rcParams['axes.labelweight'] = 'normal'  # Set axis label weight to normal
    plt.rcParams['axes.titleweight'] = 'normal'  # Set title weight to normal
    fig, ax = plt.subplots(figsize=(10, 10))  # Increased figure size
    if show_title:
        ax.set_title(f"Decision Boundary {post_fix}", fontsize=18, fontweight='bold')  # Increased font size and set to bold
    else:
        ax.set_title(f"", fontsize=18, fontweight='bold')

    # Plot background
    scatter_background = ax.scatter(
        grid_points[:, 0], grid_points[:, 1],
        c=[class_color_mapping[cls] for cls in background_predictions],
        cmap=cmap_background,
        alpha=alpha_background, s=40, label='Decision Boundary'  # Increased marker size
    )
    
    # Plot training points
    scatter_train = ax.scatter(
        X_train_2d[:, 0], X_train_2d[:, 1],
        c=[class_color_mapping[label] for label in y_train],
        edgecolor='k',
        s=marker_size_train, alpha=alpha_train, label='Training Data'
    )
    
    if not highlight_miss:
        # Plot test points
        scatter_test = ax.scatter(
            X_test_2d[:, 0], X_test_2d[:, 1],
            c=[class_color_mapping[label] for label in y_test],
            marker='X',
            edgecolor='k', s=marker_size_test, alpha=alpha_test, label='Test Data'
        )
    else:
        y_pred = classifier.predict(X_test)
        if other_classifier is not None and other_features is not None:
            other_pred = other_classifier.predict(other_features[len(X_train):])
            correct_indices = (y_test == other_pred) & (y_test != y_pred)
            if np.any(correct_indices):
                print(f"Correct indices: {correct_indices}")
            else: 
                print("No points are correctly classified by the other classifier but incorrectly classified by this one.")
            correct_label = y_test[correct_indices]

            scatter_correct_other = ax.scatter(
                X_test_2d[correct_indices, 0], X_test_2d[correct_indices, 1],
                c=[class_color_mapping[label] for label in correct_label], marker='P', edgecolor='yellow', s=marker_size_test*1.8, alpha=alpha_test,
                label='Correct by Ours, Incorrect by Baseline'
            )
        
        incorrect_indices = y_test != y_pred
        scatter_misclassified = highlight_misclassified_samples(X_test_2d, y_test, y_pred, ax, class_color_mapping, size=marker_size_test)

    if show_legend:
        # Create legends
        category_handles = [mpatches.Patch(color=class_color_mapping[cls], label=f'Class {cls}') for cls in unique_classes]
        data_type_handles = [scatter_train]
        data_type_labels = ['Training Data']
        
        if not highlight_miss:
            data_type_handles.append(scatter_test)
            data_type_labels.append('Test Data')
        else:
            data_type_handles.append(scatter_misclassified)
            data_type_labels.append('Misclassified Test Data')
            if other_classifier is not None and other_features is not None:
                data_type_handles.append(scatter_correct_other)
                data_type_labels.append('Correct by Ours, Incorrect by Baseline')

        if legend_position == 'outside':
            # Adjust the position of both legends outside the plot
            legend1 = ax.legend(handles=category_handles, title='Categories', loc='upper left', bbox_to_anchor=(1.01, 1), fontsize="18", title_fontsize="22")  # Increased font sizes
            ax.add_artist(legend1)
            ax.legend(data_type_handles, data_type_labels, title='Data Types', loc='upper left', bbox_to_anchor=(1.01, 0.87), fontsize="18", title_fontsize="22")  # Increased font sizes
            plt.subplots_adjust(right=0.75)
        else:
            # Place both legends inside the plot
            legend1 = ax.legend(handles=category_handles, title='Categories', loc='upper right', bbox_to_anchor=(0.995, 1), fontsize="18", title_fontsize="22")  # Increased font sizes
            ax.add_artist(legend1)
            ax.legend(data_type_handles, data_type_labels, title='Data Types', loc='lower right', fontsize="18", title_fontsize="22")  # Increased font sizes

    # Set axis labels and limits
    ax.set_xlabel('Projection 1', fontsize="22")  # Increased font size and set to bold
    ax.set_ylabel('Projection 2', fontsize="22")  # Increased font size and set to bold
    ax.set_xlim(min_x1-padding, max_x1+padding)
    ax.set_ylim(min_x2-padding, max_x2+padding)

    # Increase tick label font size
    ax.tick_params(axis='both', which='major', labelsize=12)  # Increased tick label size

    if save_path is not None:
        plt.savefig(save_path + f'fig_{post_fix}.pdf', bbox_inches='tight')

    # Adjust the layout
    plt.tight_layout()
    plt.show()
    plt.close()