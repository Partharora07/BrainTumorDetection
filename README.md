# BrainTumorDetection
Detection of Brain Tumor from MRI images through Deep Learning and classify the type of Tumor
# MRI Brain Tumor Detection using Deep Learning and Machine Learning

This project implements a sophisticated pipeline for detecting brain tumors from MRI images, based on the research paper: **"MRI brain tumor detection using deep learning and machine learning approaches."** The notebook leverages a combination of image processing, machine learning, and deep learning techniques to classify different types of brain tumors.

## 📖 Project Overview

The primary goal of this project is to accurately classify brain MRI images into four categories: **glioma, meningioma, pituitary tumor, and no tumor**. This is achieved through a multi-stage process that includes:

1. **Preprocessing**: Enhancing the quality of the MRI images.
2. **Segmentation**: Isolating the tumor region from the rest of the brain.
3. **Feature Extraction**: Deriving meaningful features from both the original images and the segmented tumor regions.
4. **Classification**: Using a hybrid model to classify the images based on the extracted features.

## ✨ Key Features

* **Adaptive Contrast Enhancement**: Utilizes Contrast Limited Adaptive Histogram Equalization (CLAHE) to improve the contrast of the MRI scans.
* **Fuzzy C-Means (FCM) Clustering**: Employs FCM for effective segmentation of the tumor.
* **GLCM Feature Extraction**: Extracts texture-based features using the Gray-Level Co-occurrence Matrix (GLCM).
* **CNN-based Feature Extraction**: A Convolutional Neural Network (CNN) is used to extract deep features from the images.
* **Ensemble Deep Neural Support Vector Machine (EDN-SVM)**: A simulated EDN-SVM approach, where a Support Vector Machine (SVM) classifies the fused features from both GLCM and the CNN, providing a robust classification model.

## 🛠️ Technologies & Libraries Used

This project is built using Python and relies on the following major libraries:

* **TensorFlow & Keras**: For building and training the CNN model.
* **Scikit-learn**: For the SVM classifier and performance metrics.
* **Scikit-image**: For GLCM feature extraction.
* **OpenCV**: For image preprocessing tasks like CLAHE and median filtering.
* **Matplotlib & Seaborn**: For data visualization, including plotting training history and confusion matrices.
* **NumPy**: For numerical operations.
* **fcmeans**: For Fuzzy C-Means clustering.

## ⚙️ Setup & Installation

To run this project, you'll need to set up a Python environment with the necessary libraries.

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd <your-repository-name>
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install the required packages:**
   ```bash
   pip install tensorflow scikit-learn scikit-image opencv-python matplotlib seaborn fcmeans tqdm
   ```

4. **Download the dataset**: Make sure you have the "Training" and "Testing" data folders in the root directory of the project.

## 🚀 Usage

1. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

2. Open the `MRI_Brain_Tumor_EDN_SVM_Project.ipynb` file.

3. Run the cells in the notebook sequentially to execute the entire pipeline.

## 📊 Results

The model achieves impressive results in classifying the brain tumors:

* **Multi-Class Classification Accuracy**: **80.32%**
* **Binary Classification Accuracy (Tumor vs. No Tumor)**: **93.29%**

### Visualizations

The notebook includes several visualizations to help understand the model's performance:

* **Training and Validation Accuracy/Loss Plots**: To monitor the CNN's training progress.
* **Confusion Matrices**: For both multi-class and binary classification to show the model's predictive accuracy for each class.
* **Misclassified Images**: A display of some of the images that the model failed to classify correctly, providing insight into its limitations.

## 📚 References and Citations

This project is an implementation of the following research paper and utilizes a public dataset:

* **Research Paper**: Anantharajan, S., Gunasekaran, S., Subramanian, T., & Venkatesh, R. (2024). *MRI brain tumor detection using deep learning and machine learning approaches*. Measurement: Sensors, 31, 101026.

* **Dataset**: The dataset used for training and testing is the "Brain Tumor MRI Dataset" available on Kaggle. It can be accessed here: <https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset>

