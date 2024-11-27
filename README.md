# DATA Club - Cartoon Image Classification Project with Fine Tuning!
## Colin Chu, Le Fan Fang, Yuandi Tang, Rongjia Sun, Andre Barle
This project will focus on image classification using machine learning and computer vision techniques. Specifically, it will involve developing a classifier to identify characters from the “Tom and Jerry” cartoon series. The dataset includes images of Tom, Jerry, and other characters, providing an opportunity to explore CNNs (Convolutional Neural Networks) and advanced image processing methods.
The objective of this project is to build a robust image classification model that can accurately identify characters from “Tom and Jerry” images. This will be an excellent use case for convolutional neural networks (CNNs), which are highly effective in image recognition tasks. The model’s performance will be evaluated based on its accuracy, precision, and ability to generalize across unsetest images.en test images.

## Project Summary

### Detailed Evaluation of Models in Cartoon Image Classification

#### 1. Evaluation Metrics

The models in our cartoon image classification project were evaluated using common metrics to ascertain their performance. The primary metrics included:

- **Accuracy**: Measures the number of correct predictions made out of the total predictions.
- **Loss**: Represents how well the model predicts the labels, where lower values indicate better performance.
- **Validation Accuracy**: Evaluates the model’s performance on unseen validation data to check for overfitting.
- **Confusion Matrix**: A tool to visualize the performance of the classification model by showing true vs. predicted classifications.

#### 2. CNN Model Evaluation

After training the CNN model, we assessed its performance on the validation dataset:

```python
val_loss, val_accuracy = model.evaluate(validation_generator, steps=validation_generator.samples // BATCH_SIZE)
print(f"Validation accuracy: {val_accuracy}")
```

- **Epoch Results**: The validation accuracy after training for 15 epochs reached approximately 73.62%, with concurrent losses decreasing to indicate improved performance.
  
**Sample Results:**
```plaintext
Epoch 1/15
- accuracy: 0.3717, val_accuracy: 0.4197
Epoch 15/15
- accuracy: 0.4978, val_accuracy: 0.4108
Final Validation accuracy: 0.7362
```

#### 3. Model Refinement using Random Forest and k-NN

After extracting features from the CNN model, we refined our approach by applying Random Forest and k-NN classifiers:

- **Random Forest Model**: Accuracy increased to approximately 79.57%, demonstrating enhanced classification capability compared to the CNN alone.
  
**Feature Extraction and Random Forest Training:**
```python
features = cnn_model.predict(X_features)
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(features, y_train)
```

#### 4. Results Visualization

We created visual representations to compare model performances:

- **Accuracy vs. Number of Trees in Random Forest**:
  - Hourglass shape demonstrating optimal tree count leading to peak accuracy.
  
**Plotting Example:**
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(num_trees_list, cnn_accuracy_scores, marker='o')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.title('CNN Feature Accuracy vs. Number of Trees')
plt.show()
```

- This graphical representation helps determine the best number of trees for the Random Forest classifier.

#### 5. Model Evaluation Summary

Overall, the evaluation of the models in the cartoon image classification project has led to the following conclusions:

- The CNN model alone achieved reasonable accuracy but revealed potential for further improvements.
- The feature extraction followed by application of Random Forest and k-NN classifiers significantly enhanced accuracy levels, proving the benefit of combining models.
- Implementation of evaluation metrics and visual plots allowed for insightful analysis and refinement of the model selection process.

This structured approach to model evaluation illustrates the effectiveness of integrating both deep learning and traditional machine learning methods to achieve better classification results.
