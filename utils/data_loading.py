from torch.utils.data import DataLoader, random_split


def create_data_loaders(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, batch_size=4):
    """
    Create train, validation, and test data loaders from a dataset.
    
    Args:
        dataset: The dataset to split
        train_ratio: Fraction for training data
        val_ratio: Fraction for validation data
        test_ratio: Fraction for test data
        batch_size: Batch size for data loaders
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Total size
    dataset_size = len(dataset)

    # Sizes
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size  # Remaining

    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader