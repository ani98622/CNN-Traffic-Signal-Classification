# Traffic Sings Classification Project

This project uses a deep learning model (Convolutional Neural Network) to classify German traffic signs into **43 categories**. The model is trained on the **GTSRB dataset** and is now deployed as a Streamlit app for easy access and real-time predictions.

## Try the App

Access the deployed model on Streamlit [here](https://traffic-sign-classification.streamlit.app/).

## Project Overview

### Dataset
The **German Traffic Sign Recognition Benchmark (GTSRB)** dataset includes over **50,000 images** classified into **43 traffic sign categories**. All images are resized to **30x30x3 pixels** for compatibility with the CNN.

### Key Steps

1. **Data Preparation**: 
   - Downloaded and preprocessed the **GTSRB dataset**.
   - Split the data into training and validation sets, with one-hot encoding of labels.

2. **Model Architecture**:
   - Designed a deep CNN with convolutional, pooling, Batch Normalization, and Dropout layers.
   - Configured the model to classify images into **43 categories** using a Softmax output layer.

3. **Training**:
   - Trained using the **Adam optimizer** (learning rate: **0.001**) over **30 epochs**.
   - Applied data augmentation techniques like rotation and zoom for better generalization.

4. **Evaluation**:
   - Evaluated on validation and test datasets.
   - Visualized the training process, confusion matrix, and provided a detailed classification report.

5. **Prediction**:
   - Predicted traffic signs and visualized comparisons between actual and predicted labels.

6. **Model Saving**:
   - Saved the trained model for future inference.

## Results

- The model achieved a test accuracy of **`97.5%`**.
- Performance metrics and confusion matrices provide detailed insights.

## How to Use

1. **Clone this repository**.
2. Install the required libraries.
3. Run the **Jupyter notebook** to replicate results or use the saved model for inference.

## Model

- The trained model is available for download **[here](https://github.com/ani98622/CNN-Traffic-Signal-Classification/blob/main/img_model.h5)**.

## References

- [GTSRB Dataset](https://benchmark.ini.rub.de/gtsrb_news.html)
