from django import forms
from .models import PatientRecord

class PatientForm(forms.ModelForm):
    class Meta:
        model = PatientRecord
        fields = ['patient_id', 'age', 'gender', 'viral_load', 'cd4_count', 'adherence_level']