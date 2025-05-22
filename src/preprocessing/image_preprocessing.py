import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split

def extract_faces(input_dir, output_dir, size=(128, 128)):
    """Extract faces from images and save them to output directory"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Use a pre-trained face detector (you might need to install MTCNN package)
    from mtcnn import MTCNN
    detector = MTCNN()
    
    for img_file in os.listdir(input_dir):
        if img_file.endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(input_dir, img_file)
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            faces = detector.detect_faces(img_rgb)
            
            # Extract and save each detected face
            for i, face_info in enumerate(faces):
                x, y, width, height = face_info['box']
                face = img[y:y+height, x:x+width]
                
                # Resize to uniform size
                face_resized = cv2.resize(face, size)
                
                # Save the extracted face
                output_path = os.path.join(output_dir, f"{os.path.splitext(img_file)[0]}_face_{i}.jpg")
                cv2.imwrite(output_path, face_resized)

def load_and_preprocess_images(data_dir, categories, img_size=(128, 128)):
    """Load and preprocess images from directories"""
    data = []
    labels = []
    
    for category_idx, category in enumerate(categories):
        path = os.path.join(data_dir, category)
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            try:
                img = cv2.imread(img_path)
                img = cv2.resize(img, img_size)
                img_array = img_to_array(img) / 255.0  # Normalize to [0,1]
                data.append(img_array)
                labels.append(category_idx)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    
    # Convert to numpy arrays
    X = np.array(data)
    y = np.array(labels)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Extract faces from raw images
    extract_faces("D:\\FakeDetector\\data\\image_data\\real", 
                 "D:\\FakeDetector\\data\\image_data\\processed\\real")
    extract_faces("D:\\FakeDetector\\data\\image_data\\fake", 
                 "D:\\FakeDetector\\data\\image_data\\processed\\fake")
    
    # Load and preprocess the extracted faces
    X_train, X_test, y_train, y_test = load_and_preprocess_images(
        "D:\\FakeDetector\\data\\image_data\\processed",
        ["real", "fake"],
        (128, 128)
    )
    
    # Save the preprocessed data
    np.save("D:\\FakeDetector\\data\\image_data\\X_train.npy", X_train)
    np.save("D:\\FakeDetector\\data\\image_data\\X_test.npy", X_test)
    np.save("D:\\FakeDetector\\data\\image_data\\y_train.npy", y_train)
    np.save("D:\\FakeDetector\\data\\image_data\\y_test.npy", y_test)
