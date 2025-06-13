#Data Preprocessing and Augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Initialize ImageDataGenerator for training data (with augmentation)
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,       
    rotation_range=40,          
    width_shift_range=0.2,      
    height_shift_range=0.2,     
    shear_range=0.2,            
    zoom_range=0.2,             
    horizontal_flip=True,       
    fill_mode='nearest'        
)

# For validation data, only rescaling is performed
val_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Loading the images from directories
train_generator = train_datagen.flow_from_directory(
    'path/to/fish_dataset/train',  
    target_size=(128, 128),        
    batch_size=32,                 
    class_mode='sparse'            
)

validation_generator = val_datagen.flow_from_directory(
    'path/to/fish_dataset/val',    
    target_size=(128, 128),        
    batch_size=32,
    class_mode='sparse'            
)
# Model Training
 #Train a CNN Model from Scratch
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def build_cnn_model(input_shape=(128, 128, 3), num_classes=3):
    model = Sequential()

    # Convolutional layers
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))  
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

cnn_model = build_cnn_model(input_shape=(128, 128, 3), num_classes=len(train_generator.class_indices))

history_cnn = cnn_model.fit(train_generator, epochs=10, validation_data=validation_generator)
#Experimenting with Pre-trained Models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D

# Load the pre-trained VGG16 model (without the top layers)
base_model_vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Freeze the layers in the base model
base_model_vgg16.trainable = False

# Add custom layers for our specific task
model_vgg16 = Sequential([
    base_model_vgg16,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(train_generator.class_indices), activation='softmax')
])

# Compile and train the model
model_vgg16.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_vgg16 = model_vgg16.fit(train_generator, epochs=10, validation_data=validation_generator)
 
 #Fine-Tuning Pre-trained Models
 # Unfreeze the last few layers of the base model
base_model_vgg16.trainable = True
for layer in base_model_vgg16.layers[:-4]:
    layer.trainable = False

# Recompile the model with a smaller learning rate for fine-tuning
model_vgg16.compile(optimizer=Adam(learning_rate=1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fine-tune the model
history_vgg16_fine = model_vgg16.fit(train_generator, epochs=10, validation_data=validation_generator)


#Model Evaluation
  #Evaluation Metrics

  from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = np.argmax(model.predict(X_test), axis=-1)
    report = classification_report(y_test, y_pred, target_names=train_generator.class_indices.keys())
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Classification Report for {model_name}:\n", report)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=train_generator.class_indices.keys(), yticklabels=train_generator.class_indices.keys())
    plt.title(f"Confusion Matrix: {model_name}")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

# Evaluate each model (e.g., CNN from scratch, VGG16, etc.)
evaluate_model(cnn_model, X_test, y_test, "CNN from Scratch")
evaluate_model(model_vgg16, X_test, y_test, "VGG16")

#Visualizing Training History
def plot_training_history(history, model_name):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title(f"Accuracy: {model_name}")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f"Loss: {model_name}")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Plot training history for each model
plot_training_history(history_cnn, "CNN from Scratch")
plot_training_history(history_vgg16, "VGG16")

#Deployment Using Streamlit
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load the best model (choose the one with the best performance)
model = load_model('best_fish_model.h5')

# Define function for prediction
def predict_fish(image):
    img = image.resize((128, 128))
    img_array = np.array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions)
    confidence = predictions[0][class_idx]
    return class_idx, confidence

# Streamlit UI
st.title('Fish Species Prediction')
st.write("Upload an image of a fish to classify it into a species.")

# Upload Image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Prediction
    class_idx, confidence = predict_fish(image)
    species = train_generator.class_indices.keys()[class_idx]
    
    st.write(f"Predicted Species: {species}")
    st.write(f"Model Confidence: {confidence:.2f}")





