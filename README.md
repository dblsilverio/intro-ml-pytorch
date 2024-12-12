[Introduction to Machine Learning with Pytorch - Nanodegree Projects](https://www.udacity.com/course/intro-to-machine-learning-nanodegree--nd229) 

# [Project 1 - Finding Donors for CharityML](001-supervised-learning/)

## Project Overview

The goal of this project is to accurately model individuals' incomes and identify those most likely to become donors by applying supervised learning techniques.

**Python Version**: 3.12.X - CUDA  

**Dataset Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Adult)

---

## Workflow Summary

The project follows a structured data science pipeline:
1. **Data Exploration & Preprocessing**:  
   - Investigated the structure of the dataset (e.g., categorical and continuous features).
   - Addressed skewed features using logarithmic transformations.
   - Scaled numerical features using `sklearn.preprocessing.MinMaxScaler`.
   - Applied one-hot encoding to categorical variables and converted income to binary format (<=50K as 0 and >50K as 1).
   
2. **Supervised Learning Model Evaluation**:  
   - Selected three supervised models: AdaBoost, Support Vector Machine (SVC), and Stochastic Gradient Descent Classifier (SGDClassifier).
   - Benchmarked performance on training subsets (1%, 10%, and 100% of the training dataset) using accuracy and F-beta score.

3. **Optimization**:  
   - Performed a grid search for hyperparameter tuning of the chosen AdaBoost model.
   - Compared performance improvements from the unoptimized to the optimized model.

4. **Feature Importance & Dimensionality Reduction**:  
   - Examined feature importance using AdaBoost's `feature_importances_`.  
   - Reduced the dataset to the top 5 most important features to evaluate performance trade-offs in training time, accuracy, and F-beta score.

---

## Libraries Used

The following Python libraries were utilized in the implementation:
- **Core Libraries**: `numpy`, `pandas`
- **Visualization**: `matplotlib`, `visuals.py` (Custom visualizations)
- **Scikit-learn**:
  - Preprocessing: `MinMaxScaler`
  - Supervised Learning: `AdaBoostClassifier`, `SVC`, `SGDClassifier`
  - Evaluation: `accuracy_score`, `fbeta_score`
  - Tuning: `GridSearchCV`

---

## Key Techniques

1. **Data Normalization**: Scaling continuous numerical data to [0, 1] for uniform influence.
2. **Logarithmic Transformations**: Mitigated effects of skewed distributions in `capital-gain` and `capital-loss`.
3. **One-Hot Encoding**: Converted categorical features (e.g., `workclass`, `occupation`) to numerical dummy variables.
4. **Model Selection and Comparison**: Evaluated multiple learners' performance through training time, accuracy, and F-beta score.
5. **Feature Importance Analysis**: Identified key factors influencing income predictions (e.g., capital-gain, education-num, hours-per-week).

---

## Results

### Dataset Summary:
- **Total Records**: 45,222
- **Income Distribution**:  
  - >$50,000: 24.78%  
  - ≤$50,000: 75.22%

### Model Evaluation:
Three models were evaluated for performance:

1. **AdaBoostClassifier**: Chosen as the best-performing model due to its:
   - Fast training and prediction times.
   - Superior generalization on test data with high F-beta scores.

2. **Metrics Comparison**:  

| Metric               | Unoptimized Model | Optimized Model |
|----------------------|-------------------|----------------|
| **Accuracy**         | 85.76%           | 86.67%         |
| **F-beta Score**     | 0.7246           | 0.7432         |

The tuned AdaBoost model improved both accuracy and the F-beta score on the testing dataset.

### Feature Importance Insights:
The top 5 features influencing income predictions ranked by importance:
1. `Capital-gain`
2. `Age`
3. `Education-num`
4. `Capital-loss`
5. `Hours-per-week`

Using only the top 5 predictors, the model achieved:
- **Accuracy**: 84.08% (vs. 86.67% on full dataset).
- **F-beta**: 0.6972 (vs. 0.7432 on full dataset).

# [Project 2 - CIFAR-10 Image Classifier](002-nn-with-pytorch/)

This repository contains a convolutional neural network (CNN) designed to classify images from the **CIFAR-10 dataset**. The model was implemented using PyTorch and evaluates the performance on the standard CIFAR-10 dataset after optimization. The final model achieves a test accuracy of **71.26%**, which meets the project objective of exceeding 70%.

---

## Project Overview

The **CIFAR-10 dataset** is a famous image classification dataset consisting of 60,000 color images of 32x32 resolution, divided into 10 classes:
- Airplane/Plane
- Automobile/Car
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck  

This project focuses on building a **custom neural network** to classify the dataset. By testing various architectures, transforms, and hyperparameter configurations, the final model achieves a compelling accuracy benchmark, making it useful for further experimentation and deployment purposes.

**Dataset Source**: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)

**Python Version**: 3.12.X - CUDA

---

## Model Architecture

The final neural network model is a **Convolutional Neural Network (CNN)** built with the following layers:

### Layers Overview:
1. **Convolutional Blocks**:  
   - 3 convolutional layers, each followed by batch normalization, ReLU activation, and max pooling.
   - Extract meaningful hierarchical features from the images (e.g., edges, textures, shapes).
2. **Fully Connected Layers**:  
   - 3 fully connected layers that map hierarchical features into class probabilities.
   - Dropout was added to reduce overfitting.
3. **Output Layer**:  
   - Final layer uses the **softmax activation function** to output probabilities for 10 classes.

```plaintext
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [5, 26, 30, 30]             728
       BatchNorm2d-2            [5, 26, 30, 30]              52
         MaxPool2d-3            [5, 26, 15, 15]               0
            Conv2d-4            [5, 56, 13, 13]          13,160
       BatchNorm2d-5            [5, 56, 13, 13]             112
         MaxPool2d-6              [5, 56, 6, 6]               0
            Conv2d-7             [5, 128, 4, 4]          64,640
       BatchNorm2d-8             [5, 128, 4, 4]             256
         MaxPool2d-9             [5, 128, 2, 2]               0
           Linear-10                   [5, 128]          65,664
           Linear-11                    [5, 80]          10,320
           Linear-12                    [5, 10]             810
================================================================
Total params: 155,742
Trainable params: 155,742
Non-trainable params: 0
----------------------------------------------------------------
```

### Key Design Choices
- **Batch Normalization**: Normalizes hidden layer outputs to improve convergence.
- **Dropout**: Reduces overfitting by randomly setting 35% of neuron outputs to 0.
- **Max Pooling**: Reduces spatial dimensions efficiently.

Data was preprocessed using the following PyTorch transforms:

- **Random horizontal flips** (75% probability): Augments images and introduces variability.
- **Normalization**: Standardizes pixel values to have mean 0 and variance 1, ensuring faster convergence during training.


### Summary
- **Total Parameters**: 155,742
- **Input Size**: 32x32 RGB images
- **Output**: 10 classification probabilities

---

## Results

### Training and Validation:
The **model was trained for 25 epochs**, with the following key hyperparameters:
- **Loss function**: Cross-Entropy Loss (since this is a multi-class classification problem).
- **Optimizer**: Stochastic Gradient Descent (SGD) with momentum = 0.9, learning rate = 0.001.

| Metric         | Train Loss   | Validation Accuracy | Validation Loss |
|----------------|--------------|---------------------|-----------------|
| **Epoch 1**    | 2.1801       | 43.71%             | 2.0332          |
| **Epoch 12**   | 1.8075       | 66.51%             | 1.7941          |
| **Epoch 25**   | 1.7359       | 70.68%             | 1.7530          |

### Test Accuracy:
**Final test accuracy** on test images: **71.26%**.

---

## Next Steps & Future Improvements

While this model meets the project goal of a 70% test accuracy, additional enhancements can be made to improve performance further. **Future recommendations:**

1. **Additional Convolutional Layers**:
   - Explore deeper architectures to extract more complex features.

2. **Hyperparameter Optimization**:
   - Research automated tools or manual grid search to fine-tune the learning rate, batch size, dropout probabilities, and number of filters for convolutional layers.

3. **Data Augmentation with Transforms v2**:
   - Test V2 Transforms for better ferpormance

---

## Model Evaluation

1. **Preprocessing**:
   - Dataset split into **80% training** and **20% validation**.
   - Normalized data for stabilized training.

2. **Training**:
   - Ran the `train_loader` for 25 epochs while monitoring validation accuracy.

3. **Validation**:
   - Validated after each epoch to observe generalization accuracy and loss.

4. **Testing**:
   - Evaluated the final model performance on the unseen test dataset.
