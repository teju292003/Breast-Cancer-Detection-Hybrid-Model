from django.db import models
class PredictionRecord(models.Model):
    image = models.ImageField(upload_to='uploads/')
    model_used = models.CharField(max_length=50, default='Advanced_Hybrid')
    prediction = models.CharField(max_length=20)
    confidence = models.FloatField()
    cnn_prediction = models.CharField(max_length=20, blank=True)
    cnn_confidence = models.FloatField(null=True, blank=True)
    rnn_prediction = models.CharField(max_length=20, blank=True)
    rnn_confidence = models.FloatField(null=True, blank=True)
    hybrid_prediction = models.CharField(max_length=20, blank=True)
    hybrid_confidence = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    class Meta:
        ordering = ['-created_at']
    def __str__(self):
        return f"{self.prediction} ({self.confidence:.1%}) - {self.created_at.strftime('%Y-%m-%d %H:%M')}"
