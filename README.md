# TransUNet for Chest X-Ray Segmentation

## Overview
This project implements a TransUNet model for segmenting lung regions in chest X-ray (CXR) images. The dataset consists of CXR images and corresponding left and right lung masks, which are combined for training. The model is built using TensorFlow and employs data augmentation techniques to enhance training performance.

## Dataset
The dataset contains:
- **CXR Images**: Stored in `CXR_PATH`.
- **Left Lung Masks**: Stored in `LEFT_MASK_PATH`.
- **Right Lung Masks**: Stored in `RIGHT_MASK_PATH`.

Each mask is combined to form a single segmentation mask for training.

## Installation
To run this project, install the necessary dependencies:
```bash
pip install tensorflow opencv-python numpy matplotlib scikit-learn
```

## Data Preprocessing
1. Images and masks are loaded from the specified directories.
2. Masks are combined using the `np.maximum` function.
3. Images and masks are resized to 256x256 pixels.
4. Data is normalized to a [0,1] range.
5. Data augmentation includes flipping, rotation, and zooming.

## Model Architecture
The TransUNet model consists of:
- **Encoder**: Convolutional layers with Batch Normalization and ReLU activation.
- **Bridge**: Additional convolutional layers.
- **Decoder**: Transposed convolutions with skip connections.
- **Output Layer**: 1x1 convolution with sigmoid activation.

## Training
The model is trained using:
- **Loss Function**: Binary Cross-Entropy.
- **Optimizer**: Adam with a learning rate of `1e-5`.
- **Batch Size**: 16.
- **Early Stopping**: Stops training if validation loss does not improve for 5 consecutive epochs.

## Running the Training Script
Run the following command in Google Colab or a local environment:
```python
python train.py
```

## Evaluation
Training history is visualized with accuracy and loss curves. The model's predictions are compared to ground truth masks using a helper function that plots:
- The original CXR image.
- The ground truth mask.
- The predicted mask.

## Results Visualization
To visualize predictions on the validation set:
```python
plot_predictions(model, X_val, y_val, num_samples=5)
```

## Notes
- Ensure dataset paths are correctly set before running the script.
- Augmentation techniques can be modified for better performance.

## License
This project is open-source and available for research and educational purposes.

