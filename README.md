# Multiclass-Fish-Image-Classification
AIML_Multiclass Fish Image Classification



Model Training
The models are trained using the following approach:

Data Preprocessing:

Images are resized to 128x128.
Image pixel values are rescaled to the range [0, 1].
Data augmentation techniques (rotation, zoom, flipping) are applied to increase model robustness.
Model Building:

CNN Model: A custom CNN is built and trained from scratch.
Pre-trained Models: We use pre-trained models like VGG16, ResNet50, MobileNetV2, InceptionV3, and EfficientNetB0, fine-tuning the last few layers for the fish classification task.
Model Evaluation:

Models are evaluated based on metrics such as accuracy, precision, recall, F1-score, and confusion matrix.
Training history is visualized using plots for accuracy and loss.
Model Evaluation
To evaluate the performance of the models:

Accuracy, precision, recall, F1-score, and confusion matrix are computed.
The training history for each model is plotted to show how accuracy and loss evolved over time.
The evaluation function prints out the classification report and visualizes the confusion matrix.

Deployment with Streamlit
The project is deployed using Streamlit, allowing users to upload fish images and receive predictions on the fish species.

Streamlit Application
The Streamlit app performs the following tasks:

Allows users to upload an image of a fish.
Processes the image and passes it through the trained model.
Displays the predicted species and the confidence level of the prediction.
