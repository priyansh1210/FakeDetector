import numpy as np
import tensorflow as tf
import cv2
from mtcnn import MTCNN
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import argparse

class FakeDetector:
    def __init__(self):
        # Load models
        self.news_model = tf.keras.models.load_model('D:\\FakeDetector\\models\\news_model\\best_model')
        self.image_model = tf.keras.models.load_model('D:\\FakeDetector\\models\\image_model\\best_model')
        
        # Load text vectorizer
        with open('D:\\FakeDetector\\models\\news_model\\tfidf_vectorizer.pkl', 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        # Initialize face detector
        self.face_detector = MTCNN()

    def detect_fake_news(self, text):
        """Detect if news text is fake or real"""
        # Vectorize the text
        text_vectorized = self.vectorizer.transform([text])
        
        # Predict
        prediction = self.news_model.predict(text_vectorized)[0][0]
        
        # Return result
        result = {
            'prediction': 'FAKE' if prediction > 0.5 else 'REAL',
            'confidence': float(prediction if prediction > 0.5 else 1 - prediction)
        }
        return result
    
    def detect_fake_image(self, image_path):
        """Detect if an image contains deepfake faces"""
        # Load and process image
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = self.face_detector.detect_faces(img_rgb)
        
        results = []
        for face_info in faces:
            x, y, width, height = face_info['box']
            face = img[y:y+height, x:x+width]
            
            # Resize and preprocess face
            try:
                face_resized = cv2.resize(face, (128, 128))
                face_array = img_to_array(face_resized) / 255.0
                face_batch = np.expand_dims(face_array, axis=0)
                
                # Predict
                prediction = self.image_model.predict(face_batch)[0][0]
                
                # Add result
                results.append({
                    'box': face_info['box'],
                    'prediction': 'FAKE' if prediction > 0.5 else 'REAL',
                    'confidence': float(prediction if prediction > 0.5 else 1 - prediction)
                })
            except Exception as e:
                print(f"Error processing face: {e}")
                continue
        
        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect fake news or deepfake images')
    parser.add_argument('--mode', choices=['text', 'image'], required=True,
                        help='Mode: text for fake news detection, image for deepfake detection')
    parser.add_argument('--input', required=True,
                        help='Input: text content or path to image file')
    
    args = parser.parse_args()
    
    detector = FakeDetector()
    
    if args.mode == 'text':
        # Read text from file if input is a file path
        if args.input.endswith(('.txt')):
            with open(args.input, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            text = args.input
        
        result = detector.detect_fake_news(text)
        print(f"News Detection Result: {result['prediction']} (Confidence: {result['confidence']:.2f})")
    
    elif args.mode == 'image':
        results = detector.detect_fake_image(args.input)
        
        if not results:
            print("No faces detected in the image.")
        else:
            # Load the image to draw bounding boxes
            img = cv2.imread(args.input)
            
            # Draw results on image
            for result in results:
                x, y, w, h = result['box']
                color = (0, 0, 255) if result['prediction'] == 'FAKE' else (0, 255, 0)
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                cv2.putText(img, f"{result['prediction']}: {result['confidence']:.2f}",
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Save and show result
            output_path = 'D:\\FakeDetector\\results\\detected_image.jpg'
            cv2.imwrite(output_path, img)
            print(f"Results saved to {output_path}")
            
            for i, result in enumerate(results):
                print(f"Face #{i+1}: {result['prediction']} (Confidence: {result['confidence']:.2f})")
