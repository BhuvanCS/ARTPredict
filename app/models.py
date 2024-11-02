from django.db import models
from django.contrib.auth.models import User

class AppUser(models.Model):
    user = models.OneToOneField(
        User, 
        on_delete=models.CASCADE,
        default = 1
    )
    USER_TYPE_CHOICES = (
        ('clinician', 'Clinician'),
        ('data_scientist', 'Data Scientist'),
    )
    user_type = models.CharField(max_length=50, choices=USER_TYPE_CHOICES)

    def __str__(self):
        return self.user.username

class PatientRecord(models.Model):
    patient_id = models.CharField(max_length=50, unique=True)
    age = models.PositiveIntegerField(null = True, blank = True)
    gender = models.CharField(max_length=10, null = True, blank = True)
    viral_load = models.FloatField()
    cd4_count = models.FloatField()
    adherence_level = models.CharField(max_length=50)
    sequence_data = models.TextField()
    treatment_response = models.CharField(max_length=50, default = 'Non-Responder')  

    def __str__(self):
        return f"Patient {self.patient_id}"

class PredictionResult(models.Model):
    patient = models.OneToOneField(PatientRecord, on_delete=models.CASCADE, related_name='prediction_result')
    predicted_response = models.CharField(max_length=100)
    confidence_score = models.FloatField()

    def __str__(self):
        return f"Prediction for Patient {self.patient.patient_id}"

class Interpretation(models.Model):
    patient = models.OneToOneField(PatientRecord, on_delete=models.CASCADE, related_name='interpretation')
    feature_importance = models.TextField(null=True, blank=True, help_text="Feature importance scores for the model's prediction.")
    explanation = models.TextField(blank=True, null=True, help_text="Brief explanation of the model's prediction reason.")
    lime_image = models.ImageField(upload_to='app/int_images/lime_explanations/', blank=True, null=True)
    
    def __str__(self):
        return f"Interpretation for Patient {self.patient.patient_id}"
    
class Diagnosis(models.Model):
    patient = models.ForeignKey(PatientRecord, on_delete=models.CASCADE, related_name="diagnoses")
    details = models.TextField(help_text="Detailed diagnosis or prescription for the patient.")

    def __str__(self):
        return f"Diagnosis for {self.patient.patient_id}"

class FeedbackData(models.Model):
    patient = models.ForeignKey(PatientRecord, on_delete=models.CASCADE, related_name='feedback_data')
    actual_response = models.CharField(max_length=100)
    comments = models.TextField(blank=True, null=True)

    def __str__(self):
        return f"Feedback for Patient {self.patient.patient_id}"
    