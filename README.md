# Parameterized Machine Learning Model Training with PyTorch

This Google Colab notebook provides a flexible, parameterized workflow for loading datasets and models from `torchvision`, training machine learning models, and evaluating them with robust error handling. It also supports optional early stopping to improve training efficiency.

## Overview

The notebook implements a parameterized machine learning pipeline using PyTorch, with options to:
- Select from all available datasets in `torchvision.datasets`
- Choose from a wide range of pre-trained models in `torchvision.models`
- Customize training settings such as epochs, learning rate, and batch size
- Enable optional early stopping based on validation loss with `patience` and `min_delta` parameters

The goal of this project is to provide a flexible, user-friendly, and error-resilient workflow for machine learning training in PyTorch, ideal for experimenting with different datasets and models.

## Features

- **Dataset Selection**: 
    - Users can select from any dataset available in `torchvision.datasets`.
    - If the dataset name is incorrect, the notebook provides suggestions and handles errors gracefully.

- **Model Selection**: 
    - Users can choose from popular models in `torchvision.models` (e.g., ResNet, VGG, EfficientNet).
    - If an invalid model name is provided, the notebook suggests similar models and handles errors.

- **Training**:
    - Customizable training parameters: epochs, batch size, and learning rate.
    - Uses `tqdm` progress bars to display training progress.
    - Option to enable verbose output for detailed logs of the training process.
  
- **Early Stopping**: 
    - Optionally enable early stopping based on validation loss to prevent overfitting.
    - Parameters include `patience` (number of epochs to wait for improvement) and `min_delta` (minimum improvement threshold).

- **Evaluation**:
    - Evaluates the model on a test set and prints the final accuracy and loss.

## Setup and Dependencies

The notebook is designed to run in **Google Colab**, **Jupyter Notebook** but can be adapted to other environments with minor modifications.

### Install Dependencies
Make sure you have the following dependencies installed:
- `torch`
- `torchvision`
- `tqdm`
- `matplotlib`
- `numpy`

To install dependencies in your environment, you can run:
```bash
pip install torch torchvision tqdm matplotlib numpy
```



# Dynamically get a list of all available datasets in torchvision
def get_available_datasets():
    dataset_classes = []
    for name, obj in inspect.getmembers(datasets):
        if inspect.isclass(obj) and issubclass(obj, torch.utils.data.Dataset):
            dataset_classes.append(name)
    return dataset_classes

# Dynamically get a list of all available models in torchvision
def get_available_models():
    model_classes = []
    for name, obj in inspect.getmembers(models):
        if inspect.isfunction(obj):
            model_classes.append(name)
    return model_classes

available_datasets = get_available_datasets()
available_models = get_available_models()

# Define a set of transformations, including converting images to tensors
transform = transforms.Compose([
    transforms.ToTensor()  # Convert PIL image to tensor
])

# Function to get dataset
def get_dataset(dataset_name, root='./data', batch_size=4, shuffle=True, num_workers=2):
    if dataset_name in available_datasets:
        # Dynamically fetch the dataset class from torchvision.datasets
        dataset_class = getattr(datasets, dataset_name)
        dataset_instance = dataset_class(root=root, download=True, transform=transform)
        data_loader = torch.utils.data.DataLoader(dataset_instance, 
                                                  batch_size=batch_size, 
                                                  shuffle=shuffle, 
                                                  num_workers=num_workers)
        print(f"Successfully loaded {dataset_name} dataset.")
        return data_loader
    else:
        # Handle typo in dataset name
        close_matches = difflib.get_close_matches(dataset_name, available_datasets, n=3)
        if close_matches:
            print(f"Dataset '{dataset_name}' not found. Did you mean one of these?")
            for match in close_matches:
                print(f"  - {match}")
            # Ask if the user wants to select the suggested dataset
            selected_suggestion = input(f"Would you like to load the dataset '{close_matches[0]}'? (yes/no): ").lower()
            if selected_suggestion == 'yes':
                return get_dataset(close_matches[0], root, batch_size, shuffle, num_workers)
        else:
            print(f"Dataset '{dataset_name}' not found. No close matches available.")
        return None

# Function to get model
def get_model(model_name, pretrained=True):
    if model_name in available_models:
        # Dynamically fetch the model class from torchvision.models
        model_class = getattr(models, model_name)
        model_instance = model_class(pretrained=pretrained)
        print(f"Successfully loaded {model_name} model.")
        return model_instance
    else:
        # Handle typo in model name
        close_matches = difflib.get_close_matches(model_name, available_models, n=3)
        if close_matches:
            print(f"Model '{model_name}' not found. Did you mean one of these?")
            for match in close_matches:
                print(f"  - {match}")
            # Ask if the user wants to select the suggested model
            selected_suggestion = input(f"Would you like to load the model '{close_matches[0]}'? (yes/no): ").lower()
            if selected_suggestion == 'yes':
                return get_model(close_matches[0], pretrained)
        else:
            print(f"Model '{model_name}' not found. No close matches available.")
        return None

# Training loop
def train_model(model, train_loader, criterion, optimizer, epochs, device, verbose=False):
    model = model.to(device)
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        epoch_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        
        for inputs, labels in epoch_iterator:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            epoch_iterator.set_postfix({'loss': running_loss / (epoch_iterator.n + 1)})

        if verbose:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

# Function to display a grid of images from the dataset
def show_images(images, labels):
    # Convert the images from tensors to NumPy arrays for display
    images = images.numpy()
    # Create a grid of subplots
    fig, axes = plt.subplots(1, len(images), figsize=(12, 4))
    for i, (image, label) in enumerate(zip(images, labels)):
        # Convert image tensor to HWC format for displaying
        image = np.transpose(image, (1, 2, 0))
        # Un-normalize the image if needed (optional)
        image = np.clip(image, 0, 1)  # Ensuring image is in [0, 1] range
        axes[i].imshow(image)
        axes[i].set_title(f"Label: {label}")
        axes[i].axis('off')
    plt.show()

# Main function to select dataset, model, and train
def main():
    # Dataset selection
    user_input_dataset = input("Enter a dataset name from torchvision: ")
    batch_size = int(input("Enter batch size: "))
    data_loader = get_dataset(user_input_dataset, batch_size=batch_size)

    if data_loader:
        for images, labels in data_loader:
            print(f"Loaded a batch of size: {len(images)}")
            show_images(images[:4], labels[:4])  # Show the first 4 images and their labels
            break  # Just load and display the first batch as an example
    
    # Model selection
    user_input_model = input("Enter a model name from torchvision: ")
    model = get_model(user_input_model)

    if model:
        print(f"Model {user_input_model} is ready to be used.")

    # Hyperparameters for training
    epochs = int(input("Enter the number of epochs: "))
    learning_rate = float(input("Enter the learning rate: "))
    verbose = input("Enable verbose output? (yes/no): ").lower() == 'yes'

    # Set up optimizer, loss function, and device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(model, data_loader, criterion, optimizer, epochs, device, verbose)

# Run the main function
if __name__ == '__main__':
    main()


# Parameterized Machine Learning Model Training with PyTorch

This Google Colab notebook provides a flexible, parameterized workflow for loading datasets and models from `torchvision`, training machine learning models, and evaluating them with robust error handling. It also supports optional early stopping to improve training efficiency.

## Overview

The notebook implements a parameterized machine learning pipeline using PyTorch, with options to:
- Select from all available datasets in `torchvision.datasets`
- Choose from a wide range of pre-trained models in `torchvision.models`
- Customize training settings such as epochs, learning rate, and batch size
- Enable optional early stopping based on validation loss with `patience` and `min_delta` parameters

The goal of this project is to provide a flexible, user-friendly, and error-resilient workflow for machine learning training in PyTorch, ideal for experimenting with different datasets and models.

## Features

- **Dataset Selection**: 
    - Users can select from any dataset available in `torchvision.datasets`.
    - If the dataset name is incorrect, the notebook provides suggestions and handles errors gracefully.

- **Model Selection**: 
    - Users can choose from popular models in `torchvision.models` (e.g., ResNet, VGG, EfficientNet).
    - If an invalid model name is provided, the notebook suggests similar models and handles errors.

- **Training**:
    - Customizable training parameters: epochs, batch size, and learning rate.
    - Uses `tqdm` progress bars to display training progress.
    - Option to enable verbose output for detailed logs of the training process.
  
- **Early Stopping**: 
    - Optionally enable early stopping based on validation loss to prevent overfitting.
    - Parameters include `patience` (number of epochs to wait for improvement) and `min_delta` (minimum improvement threshold).

- **Evaluation**:
    - Evaluates the model on a test set and prints the final accuracy and loss.

## Setup and Dependencies

The notebook is designed to run in **Google Colab** but can be adapted to other environments with minor modifications.

### Install Dependencies
Make sure you have the following dependencies installed:
- `torch`
- `torchvision`
- `tqdm`
- `matplotlib`
- `numpy`

To install dependencies in your environment, you can run:
```bash
pip install torch torchvision tqdm matplotlib numpy
```
### How to Use
### 1. Dataset Selection
In the notebook, you can specify the dataset using the dataset_name parameter. This allows you to choose from any available dataset in torchvision.datasets.

### 2. Model Selection
Similarly, specify the model with the model_name parameter. The notebook supports all major models available in torchvision.models.

### 3. Training
You can configure the following parameters for training:
`epochs`: Number of epochs to train the model.
`learning_rate`: Learning rate for the optimizer.
`batch_size`: Batch size for loading data.
`patience`: Number of epochs with no improvement to wait before early stopping.
`min_delta`: Minimum change in validation loss to qualify as an improvement.
`verbose`: Whether to print detailed logs for each epoch (set to yes or no).
/### 4. Early Stopping
To enable early stopping, you can set the patience and min_delta parameters. The training will stop if the validation loss does not improve for patience epochs.

### 5. Evaluation
After training, the notebook evaluates the model on the test set and prints the final accuracy and loss.
### Example input when running the notebook
```bash
Enter a dataset name from torchvision: CIFAR10
Enter batch size: 32
Enter a model name from torchvision: resnet18
Enter the number of epochs: 20
Enter the learning rate: 0.001
Enter patience for early stopping: 5
Enter minimum delta for improvement: 0.01
Enable verbose output? (yes/no): yes
```
### Output
The notebook will display the following:

Training progress with loss values.
Validation loss after each epoch.
Final accuracy and test loss after evaluation.
Display of Images
The notebook can also display the first few images from the dataset to provide a visual check of the data.

## Display of Images
The notebook can also display the first few images from the dataset to provide a visual check of the data.

## Project Structure
```bash
├── ML_Task_Colab.ipynb   # Main notebook containing the code
├── README.md             # This README file

```
## Evaluation
The notebook evaluates the model based on:

## Accuracy on the test set.
Validation loss during training for early stopping.
Error Handling
The notebook includes robust error handling with suggestions for correcting common typos in dataset names and model names.

## Future Improvements
Add more detailed logging and checkpointing during training.
Expand support for other types of datasets and models.
Implement additional training features such as learning rate schedulers.
## License
This project is open-source and available under the MIT License.


### Explanation:
- The **Overview** and **Features** sections introduce the notebook's functionality and explain the key components.
- The **Setup and Dependencies** section explains how to install necessary packages.
- The **How to Use** section gives instructions on configuring and running the notebook.
- The **Example Usage** section shows sample input/output behavior.
- The **Error Handling** section highlights the robustness in handling typos or errors in input.
- The **Future Improvements** section outlines potential areas for expanding the project.

