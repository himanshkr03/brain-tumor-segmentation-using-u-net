# NeuroMap-Tumor-Localization-Using-2D-U-Net

## 1. Introduction

Brain tumors are a significant health concern, and their accurate and timely detection is crucial for effective treatment planning and prognosis. Manual segmentation of brain tumors from medical images is time-consuming and requires significant expertise. This project aims to develop an automated brain tumor segmentation system using deep learning, specifically a 2D U-Net architecture, to assist medical professionals in this critical task.

## 2. Dataset

The project utilizes the BraTS2020 dataset, a publicly available collection of multimodal MRI scans of glioma patients. Each patient's data comprises four MRI modalities: T1, T1ce, T2, and T2-FLAIR. Expert-annotated segmentation masks are provided for each scan, delineating the tumor into sub-regions. 

**Modalities:**

- T1: Native (T1)
- T1ce: Post-contrast T1-weighted (T1Gd)
- T2: T2-weighted (T2)
- T2-FLAIR: T2 Fluid Attenuated Inversion Recovery (FLAIR)

**Segmentation Classes:**

- 0: Not Tumor (NT)
- 1: Necrotic and non-enhancing tumor core (NCR/NET)
- 2: Peritumoral edema (ED)
- 3: GD-enhancing tumor (ET) (originally labeled as 4, reassigned for continuity)


## 3. Methodology

### 3.1 Data Preprocessing

- **Dataset Split:** The BraTS2020 dataset is split into training, validation, and test sets to ensure robust model evaluation. The split ratio is approximately 60% for training, 25% for validation, and 15% for testing.
- **Data Generator:** A custom Data Generator is implemented to efficiently load and preprocess the data in batches, preventing memory overload. The generator performs the following operations:
    - Selects specific slices (60-135) from each modality to focus on the region containing the tumor.
    - Resizes images to 128x128 pixels to reduce computational complexity while preserving essential features.
    - Applies One-Hot Encoding to the segmentation masks, converting categorical labels into a numerical format suitable for model training.

### 3.2 Model Architecture

- **2D U-Net:** A 2D U-Net architecture is chosen for its effectiveness in biomedical image segmentation, particularly for segmenting small and complex regions like brain tumors. The architecture consists of an encoder path that extracts features at multiple scales and a decoder path that reconstructs the segmentation mask. Skip connections between corresponding encoder and decoder layers help preserve spatial information.

### 3.3 Training

- **Optimizer and Loss Function:** The model is compiled using the Adam optimizer and categorical cross-entropy loss function. The Adam optimizer is known for its efficiency and adaptability to various datasets. Categorical cross-entropy is a suitable loss function for multi-class segmentation problems.
- **Callbacks:** Callbacks are employed to monitor and control the training process. These include:
    - `ReduceLROnPlateau`: Dynamically adjusts the learning rate if the validation loss plateaus, preventing overfitting and aiding convergence.
    - `ModelCheckpoint`: Saves the best model weights based on validation performance, ensuring the model's optimal state is preserved.
    - `CSVLogger`: Logs training metrics to a file for later analysis and visualization.
- **Epochs:** The model is trained for 35 epochs, allowing sufficient iterations for the network to learn the underlying patterns in the data.

### 3.4 Evaluation Metrics

The model's performance is evaluated using a comprehensive set of metrics:

- **Accuracy:** Measures the overall proportion of correctly classified pixels.
- **Intersection over Union (IoU):** Quantifies the overlap between predicted and ground truth segmentations.
- **Dice Coefficient:** Evaluates the similarity between predicted and ground truth segmentations, focusing on the overlap between regions of interest.
- **Sensitivity (Recall):** Measures the proportion of true positive pixels correctly identified by the model.
- **Precision:** Measures the proportion of predicted positive pixels that are actually true positives.
- **Specificity:** Measures the proportion of true negative pixels correctly identified by the model.
- **Per-Class Dice Coefficients:** Dice coefficients are calculated separately for each tumor class (necrotic, edema, enhancing) to assess the model's performance on specific tumor regions.

## 4. Technical Specifications

- **Programming Language:** Python 3.x
- **Deep Learning Framework:** TensorFlow/Keras
- **Libraries:** NumPy, Scikit-learn, OpenCV, NiBabel, Matplotlib, Scikit-image
- **Hardware:** Google Colab environment (GPU recommended)

## 5. Results and Analysis

The trained 2D U-Net model achieves promising results in segmenting brain tumors from the BraTS2020 dataset. The model demonstrates high accuracy and Dice coefficients for most tumor classes.

**Quantitative Results (example):**

| Metric | Overall | Necrotic | Edema | Enhancing |
|---|---|---|---|---|
| Dice Coefficient | 0.85 | 0.78 | 0.89 | 0.82 |
| Sensitivity | 0.82 | 0.75 | 0.91 | 0.79 |
| Precision | 0.88 | 0.81 | 0.87 | 0.85 |

**Visualization:**

The notebook includes visualizations of the predicted segmentations alongside the ground truth masks, allowing for qualitative assessment of the model's performance. These visualizations demonstrate the model's ability to accurately delineate tumor boundaries and identify different tumor regions.

## 6. Future Scope

This project can be further extended and improved in several directions:

- **3D U-Net:** Implement a 3D U-Net architecture to leverage the full spatial context of the MRI scans, potentially leading to more accurate segmentations.
- **Hyperparameter Tuning:** Explore different hyperparameter settings to optimize the model's performance further.
- **Data Augmentation:** Apply data augmentation techniques to increase the training data diversity and improve the model's robustness.
- **Ensemble Methods:** Combine predictions from multiple models to enhance segmentation accuracy.
- **Clinical Validation:** Evaluate the model's performance on a larger and more diverse clinical dataset to assess its real-world applicability.


## 7. Installation and Usage

1. **Install Dependencies:**
2. **Download Dataset**

Follow the steps below to download and set up the BraTS2020 dataset using the Kaggle API:

---

 **Step 1: Create a Kaggle Account and Obtain API Token**
1. If you don't already have a Kaggle account, [create one here](https://www.kaggle.com/).
2. Navigate to your **Account Settings** on the Kaggle website.
3. Generate an API token by clicking on **"Create New API Token"**.
4. A `kaggle.json` file containing your API credentials will be downloaded.

---
**Step 2: Place API Token in Colab**
1. Upload the `kaggle.json` file to your Google Colab environment.
2. Use the following Python code to set up the Kaggle API:

   ```python
   import os
   from google.colab import files

   # Upload the kaggle.json file
   uploaded = files.upload()

   # Create the .kaggle directory
   os.makedirs('/root/.kaggle', exist_ok=True)

   # Move kaggle.json to the .kaggle directory
   !mv kaggle.json /root/.kaggle/

   # Set permissions
   !chmod 600 /root/.kaggle/kaggle.json

   print("Kaggle API token has been set up!")

 **Step 3: Download the BraTS2020 Dataset**
Run the following Python code to download the dataset using the Kaggle API:

```python
# Install Kaggle API if not already installed
!pip install -q kaggle

# Download the dataset
!kaggle datasets download -d awsaf49/brats2020-training-data

print("Dataset downloaded successfully!")
```
 **Step 4: Unzip the Dataset**
Unzip the downloaded dataset (`brats2020-training-data.zip`) using the following Python code:

```python
import zipfile

# Unzip the dataset
with zipfile.ZipFile("brats2020-training-data.zip", 'r') as zip_ref:
    zip_ref.extractall("brats2020")

print("Dataset unzipped successfully!")
```
3. **Run Notebook:** Execute the Jupyter notebook to train and evaluate the model.
4. **Load Trained Model:** Use `keras.models.load_model` to load the saved model for inference.
5. **Preprocess Input:** Preprocess the input MRI scans using the same steps as in the training pipeline.
6. **Predict Segmentations:** Use `model.predict` to obtain the segmentation predictions.

## 8. Contributing

Contributions to this project are welcome. Please open an issue or submit a pull request.

## 👋 HellO There! Let's Dive Into the World of Ideas 🚀

Hey, folks! I'm **Himanshu Rajak**, your friendly neighborhood tech enthusiast. When I'm not busy solving DSA problems or training models that make computers *a tad bit smarter*, you’ll find me diving deep into the realms of **Data Science**, **Machine Learning**, and **Artificial Intelligence**.  

Here’s the fun part: I’m totally obsessed with exploring **Large Language Models (LLMs)**, **Generative AI** (yes, those mind-blowing AI that can create art, text, and maybe even jokes one day 🤖), and **Quantum Computing** (because who doesn’t love qubits doing magical things?).  

But wait, there's more! I’m also super passionate about publishing research papers and sharing my nerdy findings with the world. If you’re a fellow explorer or just someone who loves discussing tech, memes, or AI breakthroughs, let’s connect!

- **LinkedIn**: [Himanshu Rajak](https://www.linkedin.com/in/himanshu-rajak-22b98221b/) (Professional vibes only 😉)
- **Medium**: [Himanshu Rajak](https://himanshusurendrarajak.medium.com/) (Where I pen my thoughts and experiments 🖋️)

Let’s team up and create something epic. Whether it’s about **generative algorithms** or **quantum wizardry**, I’m all ears—and ideas!  
🎯 Ping me, let’s innovate, and maybe grab some virtual coffee. ☕✨


