# SAR Target Classification Using Scattering Transform and Neural Networks

This project implements a Synthetic Aperture Radar (SAR) target classification pipeline using the **Scattering Transform** for feature extraction and a **Neural Network** for classification.

## 📂 Project Structure

- `project.py`: Main script to preprocess data, train the model, and evaluate it.
- `padded_imgs/`: Directory containing class-wise folders of preprocessed grayscale SAR images (64x64 px).
- `radarmodel.h5`: Trained Keras model.
- `label.pkl`: Saved LabelEncoder for class decoding.

## 🧠 Model Pipeline

1. **Image Preprocessing**:
   - Load grayscale SAR images from the specified folder structure.
   - Resize to 64x64 and normalize pixel values.

2. **Feature Extraction**:
   - Apply `Scattering2D` from the Kymatio library to extract robust features from images.

3. **Classification**:
   - Feed features into a fully connected neural network with dropout regularization.
   - Train using categorical cross-entropy loss and early stopping.

4. **Evaluation**:
   - Display accuracy/loss curves.
   - Plot confusion matrix for model predictions.

## 📈 Example Output

- Accuracy and Loss plots during training.
- Confusion matrix showing prediction performance.

## 🛠 Requirements

```bash
pip install numpy opencv-python matplotlib seaborn scikit-learn tensorflow kymatio joblib
```

## ▶️ How to Run

```bash
python project.py
```

Ensure that the `padded_imgs/` directory contains properly structured folders, each corresponding to a class label with relevant image files.

## 🧾 Output

- `radarmodel.h5`: Trained Keras model.
- `label.pkl`: Encoded labels used for prediction.
- Training metrics and plots displayed.
- Confusion matrix saved as plot.

## 📬 Author

- Developed by [Your Name]
- For academic research or SAR target recognition applications.
