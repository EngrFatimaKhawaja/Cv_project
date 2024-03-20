# Detection Faults in machines using CV
### Description
In this project, acoustic signals captured from the machines are processed and fed into the CNN model. The CNN learns to extract relevant features from the acoustic data and classify them into different categories, such as normal operation or specific fault types.

By training the CNN model on a labeled dataset containing examples of both normal machine operation and various types of faults arching,corona,losseness,tracking, the model learns to distinguish between different acoustic patterns associated with each condition.plotting the ROC curve and calculating the confusiion matrix.evaluating the model and show the 87% of accuracy.

|Sr#| Topic | 
|-|-|
|00| Overview |
|01| Dataset |
|02| Setup Instruction |
|03| Usage |
|04| Acknowledgment |
 ## Overview
 This project aims to detect faults in machines using a Convolutional Neural Network (CNN) model trained on acoustic dataset. By analyzing acoustic signals captured from machines, the CNN model can identify potential faults or anomalies, enabling proactive maintenance and minimizing downtime.
 ## Dataset
 The dataset used in this project consists of acoustic signals recorded from machines during normal operation and various fault conditions. It includes labeled 
 examples of different fault types, allowing the CNN model to learn to distinguish between normal and faulty machine behavior.
  ## Setup Instruction
Libraries: Make sure you have all the required dependencies installed. You can find the list of dependencies in the requirements.txt file.

Dataset: Download the acoustic dataset and organize it according to the provided directory structure. Ensure that the dataset is split into training, validation, and testing sets.

Training: Train the CNN model using the provided training script. Adjust hyperparameters and network architecture as needed.

Evaluation: Evaluate the trained model's performance on the validation and testing sets to assess its accuracy and generalization capability.

 ## Usage
 Training: Use the training script to train the CNN model on the acoustic dataset.
###### model_with_dropout.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))
Evaluation: Evaluate the trained model's performance on the validation or testing set.
###### test_loss, test_acc= model_with_dropout.evaluate(X_test, y_test)
###### Predict probabilities for each class
y_probabilities = model_with_dropout.predict(X_test)

###### Convert probabilities to class predictions
y_pred = np.argmax(y_probabilities, axis=1)

## Acknowledgment
This project was inspired by the need for proactive maintenance in industrial settings.

