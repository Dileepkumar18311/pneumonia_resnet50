import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# --- Configuration ---
# These paths point directly to the folders inside your Mdel directory
TRAIN_DIR = "chest_xray/train"
VAL_DIR = "chest_xray/val"
TARGET_SIZE = (224, 224)
BATCH_SIZE = 32

def get_data_generators():
    print("Setting up data generators...")
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=TARGET_SIZE, batch_size=BATCH_SIZE,
        class_mode='binary', shuffle=True
    )
    val_generator = val_datagen.flow_from_directory(
        VAL_DIR, target_size=TARGET_SIZE, batch_size=BATCH_SIZE,
        class_mode='binary', shuffle=False
    )
    return train_generator, val_generator

def build_model():
    print("Building ResNet50 model...")
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze the base model initially
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model, base_model

def main():
    train_gen, val_gen = get_data_generators()
    model, base_model = build_model()

    # --- EARLY STOPPING & CHECKPOINTING ---
    # Stops training if val_accuracy doesn't improve for 5 consecutive epochs
    # restore_best_weights ensures we keep the model state before it overfitted
    checkpoint = ModelCheckpoint("best_model.keras", monitor='val_accuracy', save_best_only=True, mode='max')
    early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

    # Phase 1: Train the custom head
    print("\n--- Phase 1: Initial Training ---")
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_gen, validation_data=val_gen, epochs=10, callbacks=[checkpoint, early_stop])

    # Phase 2: Fine-tuning the last 20 layers
    print("\n--- Phase 2: Fine-Tuning ---")
    for layer in base_model.layers[-20:]:
        layer.trainable = True

    model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_gen, validation_data=val_gen, epochs=5, callbacks=[checkpoint, early_stop])

    # Save the final model in the recommended native Keras format
    model.save("pneumonia_resnet50_model.keras")
    print("\nTraining complete! Model successfully saved to pneumonia_resnet50_model.keras")

if __name__ == "__main__":
    main()