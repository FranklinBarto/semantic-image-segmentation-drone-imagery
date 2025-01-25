# U-Net Model for Drone Imagery Segmentation

Please note the training data is dirty and has some misclassifications. The output of the model will not be usable due to the data

## Overview

This project implements a **U-Net model** for semantic segmentation of drone imagery. The U-Net architecture is a convolutional neural network designed for pixel-wise classification, making it ideal for tasks such as land cover mapping, object detection, and feature extraction in aerial images.

## Model Architecture

The U-Net model consists of the following components:

1. **Encoder (Contracting Path)**: 
   - Extracts features using convolutional layers with ReLU activation and max-pooling for down-sampling.
   - Increases feature depth at each stage.

2. **Bottleneck**:
   - Connects the encoder and decoder, providing the deepest feature representation.

3. **Decoder (Expanding Path)**:
   - Reconstructs the segmented image by up-sampling through transposed convolutions.
   - Uses skip connections to combine low-level features from the encoder for precise segmentation.

### Model Definition

The U-Net model is defined as follows:

```python
def unet_model(num_classes, input_shape):
    inputs = Input(shape=input_shape)
    # Encoder
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)

    # Decoder
    u6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

    u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model
```

### Dataset and Labeling Issues
This model was trained on a drone imagery dataset. Unfortunately, an error occurred during the labeling process, resulting in mislabeled data. Consequently, the model's output is not usable for its intended purpose as the predictions are inconsistent with the ground truth.

Key Implications of the Labeling Error:
- Misalignment of Labels: Ground truth annotations do not correspond correctly to the input images.
- Unreliable Predictions: The model’s training is skewed, leading to poor segmentation results.

### Future Improvements
To address the issues encountered, the following steps are recommended:
- Data Relabeling: Ensure accurate and consistent annotations across the dataset.
- Model Retraining: Retrain the U-Net model with corrected labels.
- Validation: Implement robust validation techniques to evaluate model performance.