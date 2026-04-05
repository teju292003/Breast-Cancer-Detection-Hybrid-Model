from django.contrib import admin
from .models import PredictionRecord
@admin.register(PredictionRecord)
class PredictionRecordAdmin(admin.ModelAdmin):
    list_display = ['prediction', 'confidence', 'model_used', 'created_at']
    list_filter = ['prediction', 'model_used']
