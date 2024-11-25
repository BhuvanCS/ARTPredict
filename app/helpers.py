from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph
from reportlab.lib.utils import Image
from reportlab.lib.units import inch
import joblib
import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def preprocess_new_data(viral_load, cd4_count, adherence_level, output_dir='app/data/preprocessed_data'):
    mean_imputer = joblib.load(os.path.join(output_dir, 'mean_imputer.pkl'))

    new_data = pd.DataFrame({
        'Viral_Load': [viral_load],
        'CD4_Count': [cd4_count],
        'Adherence_Level': [adherence_level]
    })

    # Apply mean imputation to Viral_Load and CD4_Count
    new_data = new_data.reindex(columns=['Viral_Load', 'CD4_Count', 'Adherence_Level'], fill_value=np.nan)

    new_data[['Viral_Load', 'CD4_Count']] = mean_imputer.transform(new_data[['Viral_Load', 'CD4_Count']])

    return new_data.values 

def encode_sequence(sequence):
     # One-hot encode A, T, C, G (4 bases)
    encoding_dict = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'C': [0, 0, 1, 0], 'G': [0, 0, 0, 1]}
    return np.array([encoding_dict.get(base, [0, 0, 0, 0]) for base in sequence])
    
def plot_attention_weights2(pid, attention_weights, output_dir='app/temp/'):
    os.makedirs(output_dir, exist_ok=True)
    print(attention_weights)
    attention_map = attention_weights
    attention_map = np.squeeze(attention_map) 
    attention_map = tf.nn.softmax(attention_map, axis=-1).numpy()
    attention_map_avg = np.mean(attention_map, axis=-1)

    plt.figure(figsize=(12, 2))
    plt.plot(attention_map_avg)
    plt.title("Averaged Attention Weights Across Sequence")
    plt.xlabel("Sequence Index")
    plt.ylabel("Attention Weight")
    
    attention_image_path = os.path.join(output_dir, f"attention2_sample_{pid}.png")
    plt.savefig(attention_image_path)
    plt.close()
    return attention_image_path
def plot_attention_weights(pid, attention_weights, output_dir='app/temp/'):
    os.makedirs(output_dir, exist_ok=True)
    print(attention_weights)
    attention_map = attention_weights
    attention_map = np.squeeze(attention_map) 
    attention_map = tf.nn.softmax(attention_map, axis=-1).numpy()
    print(attention_map)
    print(attention_map.shape)
    plt.figure(figsize=(20, 6))
    sns.heatmap(attention_map.T, cmap='viridis', yticklabels=False)
    
    plt.title(f'Attention Weights for Patient {pid}')
    plt.xlabel('Sequence Position')
    plt.ylabel('Attention Weight')

    
    attention_image_path = os.path.join(output_dir, f"attention_sample_{pid}.png")
    plt.savefig(attention_image_path)
    plt.close()
    return attention_image_path

def generate_pdf(response, patient, prediction, interpretation, diagnosis):
    def check_page_space(p, y, min_space):
        if y < min_space:
            p.showPage()
            y = height - 50  # Reset to top of new page
        return y
    p = canvas.Canvas(response, pagesize=A4)
    width, height = A4
    y = height - 50
    min_space = 50

    # Header
    p.setFont("Helvetica-Bold", 16)
    p.drawString(50, y, f"Diagnosis Report for Patient ID: {patient.patient_id}")
    y -= 40

    # Patient Details
    p.setFont("Helvetica", 12)
    p.drawString(50, y, f"Age: {patient.age or 'N/A'}")
    p.drawString(300, y, f"Gender: {patient.gender or 'N/A'}")
    y -= 20
    y = check_page_space(p, y, min_space)
    p.drawString(50, y, f"Viral Load: {patient.viral_load or 'N/A'}")
    p.drawString(300, y, f"CD4 Count: {patient.cd4_count or 'N/A'}")
    y -= 20
    y = check_page_space(p, y, min_space)
    p.drawString(50, y, f"Adherence Level: {patient.adherence_level or 'N/A'}")
    p.drawString(300, y, f"Strain Type: {patient.strain_type or 'N/A'}")
    y -= 20
    y = check_page_space(p, y, min_space)
    p.drawString(50, y, f"Sequence Data:")
    y -= 20
    y = check_page_space(p, y, min_space)
    style = ParagraphStyle(name="Normal", fontSize=12, leading=20)

    sequence_data = Paragraph(patient.sequence_data or 'N/A', style)
        
    sequence_data.wrap(width - 100, height) 
    sequence_data.drawOn(p, 50, y-25)
    y -= 50
    y = check_page_space(p, y, min_space)

    # Prediction Result
    p.setFont("Helvetica-Bold", 14)
    p.drawString(50, y, "Prediction:")
    y -= 20
    y = check_page_space(p, y, min_space)
    p.setFont("Helvetica", 12)
    p.drawString(50, y, f"Predicted Response: {prediction.predicted_response if prediction else patient.treatment_response}")
    y -= 20
    y = check_page_space(p, y, min_space)
    p.drawString(50, y, f"Confidence Score: {prediction.confidence_score if prediction else 'N/A'}")
    y -= 30
    y = check_page_space(p, y, min_space)

    # Interpretation Result
    p.setFont("Helvetica-Bold", 14)
    p.drawString(50, y, "Interpretation Result:")
    y -= 30
    y = check_page_space(p, y, min_space)
    p.setFont("Helvetica", 12)
    style = ParagraphStyle(name="Normal", fontSize=12, leading=20)

    explanation_paragraph = Paragraph(interpretation.explanation if interpretation else 'N/A', style)
        
    explanation_paragraph.wrap(width - 100, height) 
    explanation_paragraph.drawOn(p, 50, y-35)
    y -= 65
    y = check_page_space(p, y, min_space)

    if interpretation.lime_image:
        p.setFont("Helvetica-Bold", 14)
        p.drawString(50, y, "LIME Explanation:")
        y -= 20
        
        lime_image_path = interpretation.lime_image.path
        p.drawImage(lime_image_path, 50, y - 200, width=300, height=200)
        y -= 220 
        y = check_page_space(p, y, min_space)

    if interpretation.attention_image:
        p.setFont("Helvetica-Bold", 14)
        p.drawString(50, y, "Attention Weights Heatmap for Sequence Data:")
        y -= 20
        y = check_page_space(p, y, min_space)
        attention_image_path = interpretation.attention_image.path
        p.drawImage(attention_image_path, 50, y - 200, width=600, height=200)
        y -= 220
        y = check_page_space(p, y, min_space)

    if interpretation.attention_image2:
        p.setFont("Helvetica-Bold", 14)
        p.drawString(50, y, "Mean Attention Weights for Sequence Data:")
        y -= 20
        y = check_page_space(p, y, min_space)
        attention_image_path = interpretation.attention_image2.path
        p.drawImage(attention_image_path, 50, y - 200, width=600, height=200)
        y -= 220
        y = check_page_space(p, y, min_space)

    # Diagnosis
    p.setFont("Helvetica-Bold", 14)
    p.drawString(50, y, "Diagnosis:")
    y -= 20
    y = check_page_space(p, y, min_space)
    p.setFont("Helvetica", 12)
    p.drawString(50, y, diagnosis.details if diagnosis else 'N/A')
    y -= 30
    y = check_page_space(p, y, min_space)

    p.showPage()
    p.save()

def get_eval_metrics(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    class_report = classification_report(y_test, y_pred, target_names=['Non-Responder', 'Responder'], output_dict=True)

    return accuracy, precision, recall, f1, conf_matrix, class_report

def plot_roc_curve(y_test, y_pred_proba, roc_curve_path):
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(roc_curve_path)
    plt.close()

def plot_conf_matrix(conf_matrix, conf_matrix_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Responder', 'Responder'], yticklabels=['Non-Responder', 'Responder'], annot_kws={"size": 12})
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    plt.savefig(conf_matrix_path)
    plt.close()  