# Dataset Directory

This directory stores the face mask dataset used for training, validating, and testing the custom Convolutional Neural Network (CNN).

## Directory Structure

```text
dataset/
    with_mask/
        image001.jpg
        image002.jpg
        ...
    without_mask/
        image001.jpg
        image002.jpg
        ...
```

## Guidelines

1. **Classes**:
   - `with_mask`: Images of human faces wearing protective face masks properly.
   - `without_mask`: Images of human faces without any face mask.
2. **Supported Formats**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`.
3. **Image Quality**:
   - Minimum recommended resolution: 128x128 pixels.
   - Frontal and slightly angled face orientations.
   - Varied lighting, ethnicity, age, and background conditions for better generalization.
4. **Automated Verification**:
   - The preparation pipeline automatically validates images, discards corrupt files, and standardizes dimensions.
   - Run `python training/prepare_dataset.py --analyze` to inspect dataset distribution and counts.
