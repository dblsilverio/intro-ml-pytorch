import matplotlib.pyplot as plt


def compare_accuracies(results):
    """Plots the validation and test accuracy and compares then to the target accuracy."""

    epochs = [ix for ix, _ in enumerate(results['valid']['acc'])]
    val_accuracies = results['valid']['acc']

    test_accuracy = results['test']['acc']

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', marker='o')
    plt.axhline(y=test_accuracy, color='b', linestyle='--', label='Test Accuracy')
    plt.axhline(y=70.0, color='g', linestyle='--', label='Target Accuracy')
    plt.axhline(y=45.0, color='r', linestyle='--', label='Suboptimal Accuracy')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation vs Test Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_train_loss(results):
    """Plots the train loss."""

    epochs = [ix for ix, _ in enumerate(results['valid']['acc'])]
    train_losses = results['train']['loss']

    test_accuracy = results['test']['acc']

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss', marker='x')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train Loss')
    plt.legend()
    plt.grid(True)
    plt.show()