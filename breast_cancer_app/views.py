from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from .forms import ImageUploadForm
from .models import PredictionRecord
from .ml_models import predict_single, predict_all_models

def home(request):
    form = ImageUploadForm()
    recent = PredictionRecord.objects.all()[:5]
    return render(request, 'breast_cancer_app/home.html', {'form': form, 'recent_predictions': recent})

def predict(request):
    if request.method != 'POST':
        return redirect('breast_cancer_app:home')
    form = ImageUploadForm(request.POST, request.FILES)
    if not form.is_valid():
        messages.error(request, 'Please upload a valid image.')
        return redirect('breast_cancer_app:home')
    image = form.cleaned_data['image']
    model_choice = form.cleaned_data['model_choice']
    record = PredictionRecord(image=image, prediction='Processing', confidence=0)
    record.save()
    try:
        if model_choice == 'all':
            results = predict_all_models(record.image.path)
            record.model_used = 'All Models'
            if 'Advanced_Hybrid' in results:
                record.hybrid_prediction = results['Advanced_Hybrid']['prediction']
                record.hybrid_confidence = results['Advanced_Hybrid']['confidence']
            if 'CNN' in results:
                record.cnn_prediction = results['CNN']['prediction']
                record.cnn_confidence = results['CNN']['confidence']
            if 'RNN' in results:
                record.rnn_prediction = results['RNN']['prediction']
                record.rnn_confidence = results['RNN']['confidence']
            record.prediction = record.hybrid_prediction or 'Unknown'
            record.confidence = record.hybrid_confidence or 0.0
        else:
            pred, conf, _ = predict_single(model_choice, record.image.path)
            record.model_used = model_choice
            record.prediction = pred
            record.confidence = conf
            if model_choice == 'Advanced_Hybrid':
                record.hybrid_prediction = pred; record.hybrid_confidence = conf
            elif model_choice == 'CNN':
                record.cnn_prediction = pred; record.cnn_confidence = conf
            elif model_choice == 'RNN':
                record.rnn_prediction = pred; record.rnn_confidence = conf
        record.save()
        return redirect('breast_cancer_app:results', pk=record.pk)
    except Exception as e:
        messages.error(request, f'Error: {str(e)}')
        record.delete()
        return redirect('breast_cancer_app:home')

def results(request, pk):
    record = get_object_or_404(PredictionRecord, pk=pk)
    model_results = []
    if record.hybrid_prediction:
        model_results.append({'name': 'Hybrid (CNN+RNN)', 'prediction': record.hybrid_prediction,
                              'confidence': record.hybrid_confidence, 'is_best': True})
    if record.cnn_prediction:
        model_results.append({'name': 'CNN', 'prediction': record.cnn_prediction,
                              'confidence': record.cnn_confidence, 'is_best': False})
    if record.rnn_prediction:
        model_results.append({'name': 'RNN', 'prediction': record.rnn_prediction,
                              'confidence': record.rnn_confidence, 'is_best': False})
    return render(request, 'breast_cancer_app/results.html', {'record': record, 'model_results': model_results})

def compare_models(request):
    stats = {
        'Advanced_Hybrid': {'accuracy': 89.23, 'auc': 0.9510, 'val_acc': 91.26},
        'CNN': {'accuracy': 70.00, 'auc': 0.7583, 'val_acc': 80.58},
        'RNN': {'accuracy': 63.08, 'auc': 0.5610, 'val_acc': 69.90},
    }
    return render(request, 'breast_cancer_app/compare.html', {'stats': stats})

def about(request):
    return render(request, 'breast_cancer_app/about.html')

def prediction_history(request):
    records = PredictionRecord.objects.all()[:50]
    return render(request, 'breast_cancer_app/history.html', {'records': records})
