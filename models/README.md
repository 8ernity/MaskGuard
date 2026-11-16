# Models Directory

This directory stores the trained deep learning artifacts for the Face Mask Detection System.

## Deliverables

1. **`mask_detector.h5`**: The trained custom Convolutional Neural Network saved in standard Keras HDF5 format.
2. **`class_mapping.json`**: Explicit mapping between class indices and semantic labels:
   ```json
   {
       "0": "with_mask",
       "1": "without_mask"
   }
   ```
3. **`model_metadata.json`**: Architectural parameters, input resolution (`128x128x3`), preprocessing expectations, training timestamp, and confidence thresholds.

## Loading the Model

In Python:
```python
import tensorflow as tf

model = tf.keras.models.load_model("models/mask_detector.h5")
model.summary()
```
