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
    STRAIN_CHOICES = [
        ('HIV-1', 'HIV-1'),
        ('HIV-2', 'HIV-2'),
    ]
    patient_id = models.CharField(max_length=50, unique=True)
    age = models.PositiveIntegerField(null = True, blank = True)
    gender = models.CharField(max_length=10, null = True, blank = True)
    viral_load = models.FloatField()
    cd4_count = models.FloatField()
    adherence_level = models.FloatField(max_length=50)
    sequence_data = models.TextField()
    strain_type = models.CharField(max_length=10, choices=STRAIN_CHOICES,null = True, default = "HIV-1")
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
    explanation = models.TextField(blank=True, null=True, help_text="Brief explanation of the model's prediction reason.")
    lime_image = models.ImageField(upload_to='app/int_images/lime_explanations/', blank=True, null=True)
    attention_image = models.ImageField(upload_to='app/int_images/attention_weights/', blank=True, null=True)
    
    def __str__(self):
        return f"Interpretation for Patient {self.patient.patient_id}"
    
class Diagnosis(models.Model):
    patient = models.ForeignKey(PatientRecord, on_delete=models.CASCADE, related_name="diagnoses")
    details = models.TextField(help_text="Detailed diagnosis or prescription for the patient.")

    def __str__(self):
        return f"Diagnosis for {self.patient.patient_id}"

    