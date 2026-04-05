# Breast Cancer Detection — Django Web Application
## CNN & RNN Hybrid Model | Aditya University | Batch C14

### 🔬 Project Overview
AI-powered breast cancer detection system using:
- **Hybrid Model**: EfficientNet-B0 (CNN) + BiLSTM (RNN) — 89.23% accuracy
- **CNN Model**: 3-Block CNN from scratch — 70.00% accuracy  
- **RNN Model**: Stacked BiLSTM — 63.08% accuracy

### 📁 Project Structure
```
breast_cancer_project/
├── manage.py
├── requirements.txt
├── breast_cancer_project/
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── breast_cancer_app/
│   ├── models.py          # Database models
│   ├── views.py           # Page views
│   ├── forms.py           # Upload form
│   ├── ml_models.py       # PyTorch model loading & prediction
│   ├── urls.py            # App URLs
│   ├── admin.py           # Admin panel
│   ├── models_saved/      # Put .pth files here
│   │   ├── Advanced_Hybrid_best.pth
│   │   ├── CNN_best.pth
│   │   └── RNN_best.pth
│   └── templates/
│       └── breast_cancer_app/
│           ├── base.html
│           ├── home.html
│           ├── results.html
│           ├── compare.html
│           ├── about.html
│           └── history.html
└── media/                 # Uploaded images stored here
```

### 🚀 Setup Instructions

#### Step 1: Install dependencies
```bash
pip install -r requirements.txt
```

#### Step 2: Copy trained model weights
Copy the 3 .pth files from Google Drive (`BreastCancer_Models_PyTorch/`) to:
```
breast_cancer_app/models_saved/
```

#### Step 3: Run migrations
```bash
python manage.py makemigrations
python manage.py migrate
```

#### Step 4: Create admin user (optional)
```bash
python manage.py createsuperuser
```

#### Step 5: Run the server
```bash
python manage.py runserver
```

#### Step 6: Open in browser
```
http://127.0.0.1:8000/
```

### 📱 Pages
- **Home** (`/`) — Upload image for prediction
- **Compare** (`/compare/`) — Model performance comparison
- **History** (`/history/`) — Past prediction records
- **About** (`/about/`) — Project info & team
- **Admin** (`/admin/`) — Database management

### 🛠 Tech Stack
- **Backend**: Django 4.2+, Python 3.10+
- **Deep Learning**: PyTorch, timm (EfficientNet-B0)
- **Dataset**: BUSI (Breast Ultrasound Images)
- **Frontend**: HTML5, CSS3, JavaScript
