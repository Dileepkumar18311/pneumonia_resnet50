import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics import confusion_matrix, classification_report

# --- Configuration ---
TEST_DIR = "chest_xray/test"
TARGET_SIZE = (224, 224)
BATCH_SIZE = 32

def main():
    print("Loading test data...")
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR, target_size=TARGET_SIZE, batch_size=BATCH_SIZE,
        class_mode='binary', shuffle=False
    )

    print("Loading saved model...")
    # Loading the native Keras format we saved in train.py
    model = load_model("pneumonia_resnet50_model.keras")

    print("Evaluating model...")
    loss, accuracy = model.evaluate(test_generator)
    print(f"\nTest Accuracy: {accuracy:.4f}\n")

    print("Generating predictions...")
    preds = model.predict(test_generator)
    y_pred = np.round(preds)

    print("\n--- Confusion Matrix ---")
    print(confusion_matrix(test_generator.classes, y_pred))

    print("\n--- Classification Report ---")
    print(classification_report(test_generator.classes, y_pred))

if __name__ == "__main__":
    main()