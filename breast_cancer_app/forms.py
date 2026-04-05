from django import forms
class ImageUploadForm(forms.Form):
    image = forms.ImageField(label='Upload Breast Ultrasound Image',
        widget=forms.FileInput(attrs={'accept': 'image/*', 'class': 'file-input', 'id': 'image-upload'}))
    model_choice = forms.ChoiceField(
        choices=[('all', 'All Models (Compare)'), ('Advanced_Hybrid', 'Hybrid (CNN+RNN) — Best'),
                 ('CNN', 'CNN Only'), ('RNN', 'RNN Only')],
        initial='all', widget=forms.Select(attrs={'class': 'select-input'}))
