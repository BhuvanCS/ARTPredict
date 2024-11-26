from django.urls import path
from . import views

urlpatterns = [
    path('', views.homepage, name='homepage'),
    path('logout', views.logout_user, name='logout'),
    
    path('login/clinician', views.loginclinician, name='clinician-login'),
    path('clinician', views.clinician_dashboard, name='clinician_dashboard'),
    path('clinician/contact-us', views.contact_us_clinician, name='contact_us_clinician'),
    path('clinician/view-patient', views.view_patient, name='view_patient'),
    path('clinician/download_diagnosis/<str:patient_id>', views.download_diagnosis, name='download_diagnosis'),
    path('clinician/view-recent-patient', views.view_recent_patient, name='view_recent_patient'),
    path('clinician/view-recent-prediction', views.view_recent_prediction, name='view_recent_prediction'),
    path('clinician/write-diagnosis', views.write_diagnosis, name='write_diagnosis'),
    path('patients/add/', views.add_patient, name='add_patient'),

    path('login/ds', views.loginds, name='ds-login'),
    path('ds', views.ds_dashboard, name='ds_dashboard'),
    path('ds/contact-us', views.contact_us_ds, name='contact_us_ds'),
    path('ds/model-metrics', views.model_metrics, name='model_metrics'),
    
]