# CIFAR-10 Image Classification Project

**Project Purpose and Scope**

This project aims to perform image classification using a Deep Learning model, Convolutional Neural Network (CNN), on the popular CIFAR-10 dataset. The CIFAR-10 dataset contains 60,000 color images belonging to 10 different classes. These classes are: airplane, automobile, bird, cat, deer, dog, frog, horse, ship and truck.

**Data Set Used**

- **CIFAR-10 Data Set**: This dataset contains 60,000 images with a size of 32x32 pixels and 3 color channels. 50.000 of these images are used for training and 10.000 for testing.

**Project Steps**

1. **Data Set Loading and Preprocessing**
    - The CIFAR-10 dataset was loaded using the `load_data` function of the Keras library.
    - The dataset was divided into training and test and its dimensions were printed.
    - From the training dataset, 50 images were visualized.

2. **Data Normalization**
    - The pixel values of the images are in the range 0-255. To bring these values to the range 0-1, all pixel values were divided by 255.

3. **Converting Tags to One-Hot Encoding Format**
    - The class labels in the CIFAR-10 dataset were converted to one-hot encoded format using the `to_categorical` function.

4. **Building the Model**
    - A Sequential CNN model was created.
    - The first layer was added as a 3x3 Convolutional layer with 32 filters.
    - The second layer was added as a 3x3 Convolutional layer with 64 filters.
    - MaxPooling and Dropout layers were added to prevent the model from overfitting.
    - The data was flattened with the Flatten layer.
    - A 512 neuron Dense layer was added and a Dense layer with a softmax activation function representing 10 classes was added as the last layer.

5. **Compiling the Model**
    - The model was compiled using `categorical_crossentropy` loss function and `adam` optimizer.
    - Accuracy was used as a performance metric.

6. **Training the Model**
    - The model was trained with training data for 10 epochs and the performance of the model was evaluated using validation data at the end of each epoch.
    - Training and validation losses and accuracy values were visualized.

7. **Model Evaluation and ROC Curve**
    - After model training, predictions were taken on the test dataset.
    - ROC curve and AUC metrics were calculated and visualized.

**Key Findings**

1. **Model Performance**
    - The training and validation accuracies of the model were used to evaluate the performance of the model. Accuracy increased and loss decreased during the training process, demonstrating the model's ability to learn.
    - Visualization of training and validation losses helped to identify the tendency of the model to overfit.

2. **ROC Curve and AUC**
    - The ROC curve and AUC were used to better understand the classification performance of the model. The ROC curve evaluated the performance of the model for different classes.
    - The macro average ROC curve and AUC were calculated to evaluate the overall performance of the model.

**Significance of the Project**

This project demonstrates how image classification can be performed using deep learning methods on the CIFAR-10 dataset. Deep learning models can achieve high accuracy rates, especially on large and complex data sets. This project demonstrates how effective CNNs are in image classification tasks and offers the possibility to further analyze model performance with metrics such as the ROC curve.
