# FashionMNIST PyTorch Project

A basic PyTorch implementation for training a Convolutional Neural Network (CNN) on the FashionMNIST dataset.

## Project Structure

```
your_project_folder
├── .gitignore
├── requirements.txt
├── README.md
├── .venv
├── data
│   └── FashionMNIST
│       └── raw
│           ├── train-images-idx3-ubyte
│           ├── train-labels-idx1-ubyte
│           ├── t10k-images-idx3-ubyte
│           └── t10k-labels-idx1-ubyte
└── main.py
```

## Setup

1. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset

The FashionMNIST dataset contains 70,000 grayscale images in 10 categories:
- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

**Training set:** 60,000 images  
**Test set:** 10,000 images  
**Image size:** 28x28 pixels

## Model Architecture

The CNN model consists of:
- 3 Convolutional layers (32, 64, 128 filters)
- MaxPooling layers
- 3 Fully connected layers (256, 128, 10 neurons)
- Dropout for regularization
- ReLU activation functions

## Training

Run the training script:
```bash
python main.py
```

### Hyperparameters
- **Batch size:** 64
- **Learning rate:** 0.001
- **Epochs:** 10
- **Optimizer:** Adam
- **Loss function:** CrossEntropyLoss

## Output

The trained model will be saved as `fashion_mnist_model.pth`.

## Results

Expected test accuracy: ~90-92% after 10 epochs

## GPU Support

The code automatically detects and uses GPU if available (CUDA). Otherwise, it falls back to CPU.

## Customization

You can modify the hyperparameters in `main.py`:
```python
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10
```

## License

This project is open source and available under the MIT License.