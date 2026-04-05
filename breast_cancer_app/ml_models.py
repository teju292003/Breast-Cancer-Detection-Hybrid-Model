import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
import os
from django.conf import settings

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        self.backbone = timm.create_model('efficientnet_b0', pretrained=False)
        self.backbone.global_pool = nn.Identity()
        self.backbone.classifier = nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.lstm = nn.LSTM(input_size=1280, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)
        self.bn = nn.BatchNorm1d(256)
        self.fc1 = nn.Linear(256, 256); self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 64); self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 1)
    def forward(self, x):
        x = self.normalize(x)
        f = self.backbone.forward_features(x); f = self.pool(f)
        B, C, H, W = f.shape; f = f.view(B, C, H*W).permute(0, 2, 1)
        o, _ = self.lstm(f); o = o[:, -1, :]
        x = self.bn(o); x = torch.relu(self.fc1(x)); x = self.dropout1(x)
        x = torch.relu(self.fc2(x)); x = self.dropout2(x)
        return self.fc3(x)

class RNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=224*3, hidden_size=128, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(0.3)
        self.lstm2 = nn.LSTM(input_size=256, hidden_size=64, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(128, 128); self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 1)
    def forward(self, x):
        B = x.size(0); x = x.permute(0, 2, 3, 1).reshape(B, 224, 224*3)
        x, _ = self.lstm1(x); x = self.dropout1(x)
        x, _ = self.lstm2(x); x = x[:, -1, :]
        x = torch.relu(self.fc1(x)); x = self.dropout2(x)
        return self.fc2(x)

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1))
    def forward(self, x):
        return self.classifier(self.features(x))

_loaded_models = {}

def get_model(model_name):
    if model_name in _loaded_models:
        return _loaded_models[model_name]
    model_map = {
        'Advanced_Hybrid': (HybridModel, 'Advanced_Hybrid_best.pth'),
        'RNN': (RNNModel, 'RNN_best.pth'),
        'CNN': (CNNModel, 'CNN_best.pth'),
    }
    ModelClass, filename = model_map[model_name]
    model_path = os.path.join(settings.MODEL_DIR, filename)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}. Copy .pth files from Google Drive to breast_cancer_app/models_saved/")
    model = ModelClass()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device); model.eval()
    _loaded_models[model_name] = model
    return model

def preprocess_image(image_path):
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    img = Image.open(image_path).convert('RGB')
    return transform(img).unsqueeze(0).to(device)

def predict_single(model_name, image_path):
    model = get_model(model_name)
    img_tensor = preprocess_image(image_path)
    with torch.no_grad():
        output = model(img_tensor).squeeze()
        prob = torch.sigmoid(output).item()
    prediction = 'Malignant' if prob > 0.5 else 'Benign'
    confidence = prob if prob > 0.5 else 1 - prob
    return prediction, confidence * 100, prob

def predict_all_models(image_path):
    results = {}
    for name in ['Advanced_Hybrid', 'RNN', 'CNN']:
        try:
            pred, conf, raw = predict_single(name, image_path)
            results[name] = {'prediction': pred, 'confidence': conf, 'raw_prob': raw}
        except FileNotFoundError:
            results[name] = {'prediction': 'Not loaded', 'confidence': 0.0, 'raw_prob': 0.5}
    return results
