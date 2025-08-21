import os
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2

# Load MobileNetV2 pretrained on ImageNet
model = MobileNetV2(weights="imagenet", include_top=True)
print("Pretrained MobileNetV2 loaded successfully.")

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    'kiwi_images',            # folder with your images
    target_size=(224,224),    # resize images to 224,224
    batch_size=8,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    'kiwi_images',
    target_size=(224,224),
    batch_size=8,
    class_mode='categorical',
    subset='validation'
)

# Number of classes (just kiwi for now)
num_classes = train_generator.num_classes  # this will detect how many folders/classes you have

# Remove the last layer of MobileNetV2
base_model = Model(inputs=model.input, outputs=model.layers[-2].output)

# Add a new Dense layer for your classes
new_predictions = Dense(num_classes, activation='softmax')(base_model.output)

# Create new model
new_model = Model(inputs=base_model.input, outputs=new_predictions)

print(f"Modified MobileNetV2 output layer for {num_classes} classes.")

# Compile the new model
new_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),  # low learning rate for fine-tuning
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
new_model.fit(
    train_generator,
    epochs=5,  # small demo, you can increase later
    validation_data=val_generator
)

# Save the fine-tuned model
new_model.save("kiwi.h5")
print("Fine-tuned model saved as kiwi.h5")

from tensorflow.keras.models import load_model
model = load_model("kiwi.h5")
