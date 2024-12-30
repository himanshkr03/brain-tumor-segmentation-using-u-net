# Brain Tumor Segmentation using U-Net

## Project Overview

This project focuses on the segmentation of brain tumors from multimodal Magnetic Resonance Imaging (MRI) scans using a 2D U-Net architecture. The project leverages the BraTS2020 dataset, which contains MRI scans with expert-annotated segmentation masks delineating various tumor sub-regions.

## Dataset

The project utilizes the BraTS2020 dataset, which includes multimodal MRI scans of glioma patients. Each patient's data comprises four MRI modalities: T1, T1ce, T2, and T2-FLAIR. Expert-annotated segmentation masks are provided for each scan, delineating the tumor into sub-regions.

- **Modalities:** T1, T1ce, T2, T2-FLAIR
- **Segmentation Classes:**
    - 0: Not Tumor (NT)
    - 1: Necrotic and non-enhancing tumor core (NCR/NET)
    - 2: Peritumoral edema (ED)
    - 3: Missing (No pixels in all the volumes contain label 3) - Replaced with label 4
    - 4: GD-enhancing tumor (ET) - Reassigned as label 3

## Methodology

1. **Data Preprocessing:**
   - The dataset is split into training, validation, and test sets.
   - A Data Generator is used to load and preprocess the data, including:
     - Selecting specific slices (60-135) from each modality.
     - Resizing images to 128x128 pixels.
     - Applying One-Hot Encoding to the segmentation masks.

2. **Model Architecture:**
   - A 2D U-Net architecture is employed for segmentation.
   - The model is compiled with the Adam optimizer and categorical cross-entropy loss function.

3. **Training:**
   - The model is trained for 35 epochs.
   - Callbacks are used for learning rate scheduling and model checkpointing.

4. **Evaluation:**
   - The model's performance is evaluated using metrics such as accuracy, Intersection over Union (IoU), Dice coefficient, sensitivity, precision, and specificity.

## Results

The trained model achieves promising results in segmenting brain tumors from MRI scans. Detailed evaluation metrics are presented in the notebook.

## Usage

To use the trained model:

1. Load the saved model using `keras.models.load_model`.
2. Preprocess the input MRI scans using the same steps as in the training pipeline.
3. Use the `model.predict` method to obtain the segmentation predictions.

## Dependencies

- Python 3.x
- TensorFlow/Keras
- NumPy
- Scikit-learn
- OpenCV
- NiBabel
- Matplotlib
- Scikit-image

## Installation

1. Install the required dependencies using `pip`:

2. Download the BraTS2020 dataset.

3. Run the Jupyter notebook to train and evaluate the model.

## Contributing

Contributions to this project are welcome. Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.
