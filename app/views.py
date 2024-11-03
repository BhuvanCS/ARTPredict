from django.shortcuts import render, redirect
from .models import PatientRecord, PredictionResult, Interpretation, FeedbackData, AppUser, Diagnosis
from .forms import PatientForm
from .helpers import generate_pdf, preprocess_new_data, get_eval_metrics, plot_roc_curve, plot_conf_matrix
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import joblib
from lime.lime_tabular import LimeTabularExplainer
import numpy as np
from django.core.files import File
import matplotlib
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import load_model



matplotlib.use('Agg')

model = joblib.load('app/models/random_forest_model.pkl')
nn_model = load_model('app/models/neural_network_model.h5')

def homepage(request):
    return render(request, 'homepage.html')

def loginds(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is None:
            return render(request, 'login_ds.html', {'error': 'Invalid credentials'})
        else:
            u = User.objects.get(username=username)
            user_type = AppUser.objects.get(user=u).user_type
            if  user_type == 'data_scientist':
                login(request, user)
                return redirect('ds_dashboard')
            else:
                return render(request, 'login_ds.html', {'error': 'Invalid user type.'})

    return render(request, 'login_ds.html')

def loginclinician(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is None:
            return render(request, 'login_clinician.html', {'error': 'Invalid credentials'})
        else:
            u = User.objects.get(username=username)
            user_type = AppUser.objects.get(user=u).user_type
            if  user_type == 'clinician':
                login(request, user)
                return redirect('clinician_dashboard') 
            else:
                return render(request, 'login_clinician.html', {'error': 'Invalid user type.'})

    return render(request, 'login_clinician.html')

def clinician_dashboard(request):
    return render(request, 'clinician_dashboard.html')

def ds_dashboard(request):
    return render(request, 'ds_dashboard.html')

def view_patient(request):
    if request.method == 'POST':
        patient_id = request.POST['patient_id']
        try:
            patient = PatientRecord.objects.get(patient_id=patient_id)
        except PatientRecord.DoesNotExist:
            patient = None
        return render(request, 'view_patient.html', {'patient': patient})
    return render(request, 'view_patient.html')

def download_diagnosis(request, patient_id):
    try:
        patient = PatientRecord.objects.get(patient_id=patient_id)
        prediction = PredictionResult.objects.filter(patient=patient).first() or None
        interpretation = Interpretation.objects.filter(patient=patient).first() or None
        diagnosis = Diagnosis.objects.filter(patient=patient).first() or None
        response = HttpResponse(content_type='text/plain')
        response['Content-Disposition'] = f'attachment; filename="diagnosis_{patient_id}.pdf"'

        generate_pdf(response, patient, prediction, interpretation, diagnosis)
        return response
    except PatientRecord.DoesNotExist:
        return HttpResponse("Patient not found", status=404)
    except (PredictionResult.DoesNotExist, Interpretation.DoesNotExist, Diagnosis.DoesNotExist):
        return HttpResponse("Required information is missing", status=404)
    

    
def patient_list(request):
    patients = PatientRecord.objects.all()
    return render(request, 'patient_list.html', {'patients': patients})

def add_patient(request):
    if request.method == 'POST':
        form = PatientForm(request.POST)
        if form.is_valid():
            form.save()
            global_x_train = np.load('app/models/x_train.npy')
            global_x_train_nn = np.load('app/models/x_train_nn.npy')
            viral_load = float(request.POST['viral_load'])
            cd4_count = float(request.POST['cd4_count'])
            adherence_level = float(request.POST['adherence_level'])

            viral_load, cd4_count, adherence_level = preprocess_new_data(viral_load, cd4_count, adherence_level)[0]
            
            input_data = np.array([[viral_load, cd4_count, adherence_level]])

            #predicted_response = model.predict(input_data)[0]
            predicted_response = nn_model.predict(input_data)[0]
            print(predicted_response)
            #predicted_response = 'Responder' if predicted_response == 1 else 'Non-Responder'
            predicted_response = 'Responder' if predicted_response[0] > 0.45 else 'Non-Responder'
            confidence_score = max(model.predict_proba(input_data)[0])
            print(predicted_response)

            patient = PatientRecord.objects.get(patient_id=request.POST['patient_id'])

            # LIME interpretation
            explainer = LimeTabularExplainer(
                training_data=global_x_train_nn,
                #training_data=global_x_train,
                feature_names=['Viral_Load', 'CD4_Count', 'Adherence_Level'],
                mode='classification'
            )

            def predict_fn(data):
                probabilities = nn_model.predict(data)
                # For binary classification, probabilities will be of shape (n_samples, 1)
                return np.hstack((1 - probabilities, probabilities))  # Shape (n_samples, 2)
            
            print(input_data[0])
            explanation = explainer.explain_instance(
                data_row=input_data[0],
                #predict_fn=model.predict_proba
                predict_fn=predict_fn
            )

            explanation_text = explanation.as_list()
            
            patient.treatment_response = predicted_response
            patient.save()

            PredictionResult.objects.create(
                patient = patient,
                predicted_response = predicted_response,
                confidence_score = confidence_score
            )

            Interpretation.objects.create(
                patient = patient,
                feature_importance = explanation_text,
                explanation = explanation_text
            )

            
            fig = explanation.as_pyplot_figure()
            fig.suptitle(f"LIME Explanation for Patient ID: {patient.patient_id}")

            
            temp_image_path = f"app/temp/lime_explanation_{patient.patient_id}.png"
            fig.savefig(temp_image_path, bbox_inches='tight')
            plt.close(fig)  

            
            interpretation = Interpretation.objects.get(patient=patient)
            with open(temp_image_path, 'rb') as image_file:
                interpretation.lime_image.save(f'lime_explanation_{patient.patient_id}.png', File(image_file), save=True)

            os.remove(temp_image_path)

            return redirect('add_patient')
    else:
        form = PatientForm()
    return render(request, 'add_patient.html', {'form': form, 'google_form_url': 'https://forms.gle/dbVkR3aJ5kmn6qXg7'})

def view_recent_patient(request):
    patient = PatientRecord.objects.latest('id') 
    data = {
        "patient_id": patient.patient_id,
        "age": patient.age,
        "gender": patient.gender,
        "viral_load": patient.viral_load,
        "cd4_count": patient.cd4_count,
        "adherence_level": patient.adherence_level,
        "sequence_data": patient.sequence_data
    }
    return JsonResponse(data)

def view_recent_prediction(request):
    patient = PatientRecord.objects.latest('id')
    prediction = PredictionResult.objects.get(patient=patient)
    interpretation = Interpretation.objects.get(patient=patient)
    data = {
        "patient_id": patient.patient_id,
        "predicted_response": prediction.predicted_response,
        "confidence_score": prediction.confidence_score, 
        "feature_importance": interpretation.explanation,
        "explanation": interpretation.explanation
    }
    return JsonResponse(data)

@csrf_exempt
def write_diagnosis(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        diagnosis_text = data.get('diagnosis')
        patient = PatientRecord.objects.latest('id')
        Diagnosis.objects.create(patient=patient, details=diagnosis_text)
        return JsonResponse({"message": "Diagnosis saved successfully."})
    return JsonResponse({"error": "Invalid request"}, status=400)

def model_metrics(request):
    model_path = 'app/models/neural_network_model.h5'
    x_test_path = 'app/models/x_test_nn.npy'
    y_test_path = 'app/models/y_test_nn.npy'
    
    if os.path.exists(model_path) and os.path.exists(x_test_path) and os.path.exists(y_test_path):
        model = load_model(model_path)
        X_test = np.load(x_test_path)
        y_test = np.load(y_test_path)
        
        y_pred_proba = model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()  # Convert probabilities to binary predictions
        
        accuracy, precision, recall, f1, conf_matrix, class_report = get_eval_metrics(y_test, y_pred)

        print(class_report)

        # Plot confusion matrix
        conf_matrix_path = 'app/static/images/confusion_matrix.png'
        plot_conf_matrix(conf_matrix, conf_matrix_path)
        
        # ROC Curve
        roc_curve_path = 'app/static/images/roc_curve.png'
        plot_roc_curve(y_test, y_pred_proba, roc_curve_path)

        context = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'class_report': class_report,
            'conf_matrix_path': conf_matrix_path,
            'roc_curve_path': roc_curve_path,
        }
        return render(request, 'model_metrics.html', context)
    else:
        return render(request, 'model_metrics.html', {'error_message': 'Model or test data not found.'})

def logout_user(request):
    logout(request)
    return redirect('homepage')
