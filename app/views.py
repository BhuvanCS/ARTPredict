from django.shortcuts import render, redirect
from .models import PatientRecord, PredictionResult, Interpretation, AppUser, Diagnosis
from .forms import PatientForm
from .helpers import generate_pdf,  encode_sequence, plot_attention_weights, get_eval_metrics, plot_roc_curve, plot_conf_matrix
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
from tensorflow.keras.preprocessing.sequence import pad_sequences
import shap



matplotlib.use('Agg')

nn_model = load_model('app/models/neural_network_model.h5')
strain_encoder = joblib.load('app/data/preprocessed_data/encoder_strain.pkl') 
response_encoder = joblib.load('app/data/preprocessed_data/encoder_response.pkl')
scaler = joblib.load('app/data/preprocessed_data/scaler.pkl')

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
            X_train_seq = np.load('app/models/x_train_seq_nn.npy')
            X_train_num = np.load('app/models/x_train_nn.npy')
            viral_load = float(request.POST['viral_load'])
            cd4_count = float(request.POST['cd4_count'])
            adherence_level = float(request.POST['adherence_level'])
            sequence_data = request.POST['sequence_data'] 
            strain_type = request.POST['strain_type']

            patient = PatientRecord.objects.get(patient_id=request.POST['patient_id'])

            sequence_data_encoded = encode_sequence(sequence_data)
            #sequence_data_encoded = pad_sequences(sequence_data_encoded, maxlen=400, padding='post', truncating='post', dtype='float32')
            sequence_data_encoded = sequence_data_encoded.reshape((1, 400, 1))  # (batch_size, sequence_length, num_features)
            print(sequence_data_encoded.shape)
            strain_type_encoded = strain_encoder.transform([strain_type])[0]

            numerical_features = np.array([[viral_load, cd4_count, adherence_level, strain_type_encoded]])
            numerical_features = scaler.transform(numerical_features)
            prediction, attention_weights = nn_model.predict([numerical_features, sequence_data_encoded])

            predicted_class = np.argmax(prediction, axis=1)
            predicted_label = response_encoder.inverse_transform(predicted_class)[0]  # Decode the prediction
            confidence_score = prediction[0][predicted_class[0]]
            patient.treatment_response = predicted_label
            patient.save()
            print(predicted_label, predicted_class, confidence_score)
            explainer = LimeTabularExplainer(
                X_train_num,  # Your training data for the numerical features
                feature_names=['Viral_Load', 'CD4_Count', 'Adherence_Level', 'Strain_Type', 'Sequence_1'],  # Names of the numerical features
                class_names=['Non-Responder', 'Responder'],  # Your class names
                discretize_continuous=True
            )

            def predict_fn(input):
                print("Predict_fn called with input shape:", input.shape)
                predictions = []

                for sample in input:
                    numerical_data = np.array(sample).reshape(1, -1)
                    prediction, _ = nn_model.predict([numerical_data, sequence_data_encoded.reshape(1,400,1)])
                    predictions.append(prediction[0])
                
                output =  np.array(predictions)
                print("Predict_fn output shape:", output.shape)
                return output
            lime_exp = explainer.explain_instance(numerical_features[0], predict_fn=predict_fn, num_samples=50)
            explanation_text = lime_exp.as_list()
            
            fig = lime_exp.as_pyplot_figure()
            fig.suptitle(f"LIME Explanation for Patient ID: {patient.patient_id}")
            
            temp_image_path = f"app/temp/lime_explanation_{patient.patient_id}.png"
            fig.savefig(temp_image_path, bbox_inches='tight')
            plt.close(fig)

            attention_image_path = plot_attention_weights(patient.patient_id, attention_weights)

            PredictionResult.objects.create(
                patient = patient,
                predicted_response = predicted_label,
                confidence_score = confidence_score
            )

            Interpretation.objects.create(
                patient = patient,
                explanation = explanation_text
            )
            interpretation = Interpretation.objects.get(patient=patient)
            with open(temp_image_path, 'rb') as image_file:
                interpretation.lime_image.save(f'lime_explanation_{patient.patient_id}.png', File(image_file), save=True)
                
            with open(attention_image_path, 'rb') as image_file:
                interpretation.attention_image.save(f'attention_mechanism_{patient.patient_id}.png', File(image_file), save=True)
            os.remove(temp_image_path)
            os.remove(attention_image_path)

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
    x_test_num_path = 'app/models/x_test_nn.npy'
    x_test_seq_path = 'app/models/x_test_seq_nn.npy'
    y_test_path = 'app/models/y_test_nn.npy'
    
    if os.path.exists(model_path) and os.path.exists(x_test_num_path) and os.path.exists(x_test_seq_path) and os.path.exists(y_test_path):
        model = load_model(model_path)
        X_test_num = np.load(x_test_num_path)
        X_test_seq = np.load(x_test_seq_path)
        y_test = np.load(y_test_path)
        
        y_pred_proba = model.predict([X_test_num, X_test_seq])
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_test_labels = np.argmax(y_test, axis=1)
        
        accuracy, precision, recall, f1, conf_matrix, class_report = get_eval_metrics(y_test_labels, y_pred)

        print(class_report)

        # Plot confusion matrix
        conf_matrix_path = 'app/static/images/confusion_matrix.png'
        plot_conf_matrix(conf_matrix, conf_matrix_path)
        
        # ROC Curve
        roc_curve_path = 'app/static/images/roc_curve.png'
        plot_roc_curve(y_test_labels, y_pred, roc_curve_path)

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
