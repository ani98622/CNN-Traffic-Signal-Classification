

# Traffic-Signal-Classification Project

This project implements a deep learning model for recognizing German traffic signs using a Convolutional Neural Network (CNN). The model is trained on the **GTSRB dataset**, which contains **43 different classes** of traffic signs. The implementation includes data preprocessing, model creation, training, evaluation, and predictions on test data. Additionally, the project visualizes the data and the model's performance.

## Project Overview

### Dataset
The **German Traffic Sign Recognition Benchmark (GTSRB)** dataset is utilized for this project. The dataset includes over **50,000 images** classified into **43 categories** of traffic signs. Each image is resized to **30x30x3 pixels** to standardize input to the CNN.

### Key Steps

1. **Data Preparation**: 
   - Downloaded and uncompressed the **GTSRB dataset**.
   - Visualized the distribution of images across different classes.
   - Preprocessed the images and split the data into training and validation sets.
   - Applied one-hot encoding to the labels for model compatibility.

2. **Model Architecture**:
   - Designed a deep Convolutional Neural Network (CNN) with multiple convolutional and pooling layers.
   - Incorporated Batch Normalization and Dropout layers to enhance model performance and reduce overfitting.
   - Configured the model to classify images into **43 categories** using a Softmax output layer.

3. **Training**:
   - Utilized the **Adam optimizer** with a learning rate of **0.001**.
   - Trained the model over **30 epochs** with data augmentation techniques such as rotation, zoom, and shifts to improve generalization.
   - Monitored the model's performance using validation data.

4. **Evaluation & Visualization**:
   - Evaluated the model's performance on validation and test datasets.
   - Visualized the training history and the loss over epochs to understand the model's learning process.
   - Generated a **confusion matrix** to analyze the model's accuracy across different traffic sign classes.
   - Provided a detailed classification report.

5. **Prediction**:
   - Applied the trained model to predict traffic signs on test images.
   - Visualized predictions with a comparison between actual and predicted labels, highlighting correct and incorrect predictions.

6. **Model Saving**:
   - Saved the trained model for future inference and deployment.

## Results

- The model achieved a test accuracy of over **`97.5%`**.
- Detailed performance metrics and confusion matrices are included to provide insights into the model's classification capabilities.

## Visualizations

- Data distribution, sample images, and model performance visualizations are provided to enhance the understanding of the project's workflow and results.

## How to Use

1. **Clone this repository**.
2. Ensure you have all the required libraries installed.
3. Run the **Jupyter notebook** to replicate the results or use the saved model for inference.

## Model

- The trained model is available for download **[here](https://github.com/ani98622/CNN-Traffic-Signal-Classification/blob/main/img_model.h5)**.

## References

- [GTSRB Dataset](https://benchmark.ini.rub.de/gtsrb_news.html)
