import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Define the CNN model with more layers to capture spatial hierarchies
def create_cnn_model():
    #includes resizing, normalization, and augmentation to enhance the robustness of the model.
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Assuming binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


#path to Train data is local folder :- 

traindata_dir = 'C:/JatinkRai/patternmatch/projects/UpdatedCode/datasets/images/' #'C:/JatinkRai/patternmatch/projects/UpdatedCode/datasets/Images/Training_Set/Training_Set/Training/' #'C:/JatinkRai/patternmatch/projects/UpdatedCode/datasets/images/'

# Data preparation
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    traindata_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

# Model training with error handling
try:
    trainmodeloutput_dir = 'C:\\JatinkRai\\patternmatch\\projects\\UpdatedCode\\ModelCNN\\CNNModelOutput'
    trainmodeloutput_filename = '\\retinal_cnn_model_SIUC.h5'

    Trainmodel_fileSavePath = trainmodeloutput_dir + trainmodeloutput_filename
    model = create_cnn_model()
    history = model.fit(train_generator, epochs=5)
    model.save(Trainmodel_fileSavePath)
except Exception as e:
    print(f"An error occurred during model training: {e}")
else:
    print("Model trained and saved successfully.")
finally:
    print("Model training process is complete.")
