from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import PatientRecord, PredictionResult, Interpretation, FeedbackData, AppUser, Diagnosis


admin.site.register(PatientRecord)
admin.site.register(PredictionResult)
admin.site.register(Interpretation)
admin.site.register(FeedbackData)
admin.site.register(Diagnosis)
admin.site.register(AppUser)
