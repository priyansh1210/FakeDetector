import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model_path, X_test, y_test, model_type):
    """Evaluate a model and save the results"""
    # Load the model
    model = tf.keras.models.load_model(model_path)
    
    # Predict on test data
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    # Calculate metrics
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot and save confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Fake'], 
                yticklabels=['Real', 'Fake'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_type} Model')
    plt.savefig(f'D:\\FakeDetector\\results\\{model_type}_confusion_matrix.png')
    
    # Save metrics to file
    with open(f'D:\\FakeDetector\\results\\{model_type}_evaluation.txt', 'w') as f:
        f.write(f"Model Evaluation Results for {model_type} Model\n")
        f.write("="*50 + "\n")
        f.write(f"Accuracy: {report['accuracy']:.4f}\n")
        f.write(f"Precision (Fake): {report['1']['precision']:.4f}\n")
        f.write(f"Recall (Fake): {report['1']['recall']:.4f}\n")
        f.write(f"F1-Score (Fake): {report['1']['f1-score']:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred))
    
    return report

# Evaluate news model
X_test_news = np.load('D:\\FakeDetector\\data\\news_data\\tfidf_test.npy')
y_test_news = np.load('D:\\FakeDetector\\data\\news_data\\y_test.npy')
news_report = evaluate_model(
    'D:\\FakeDetector\\models\\news_model\\best_model',
    X_test_news, 
    y_test_news, 
    'news'
)

# Evaluate image model
X_test_image = np.load('D:\\FakeDetector\\data\\image_data\\X_test.npy')
y_test_image = np.load('D:\\FakeDetector\\data\\image_data\\y_test.npy')
image_report = evaluate_model(
    'D:\\FakeDetector\\models\\image_model\\best_model',
    X_test_image, 
    y_test_image, 
    'image'
)

print("Evaluation complete! Results saved to D:\\FakeDetector\\results\\")
