from django.conf import settings
from django.db import models

class Machine(models.Model):
    name = models.CharField(max_length=120)
    model_number = models.CharField(max_length=120, unique=True)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    def __str__(self): return f"{self.name} ({self.model_number})"

class Manual(models.Model):
    machine = models.ForeignKey(Machine, on_delete=models.CASCADE, related_name="manuals")
    title = models.CharField(max_length=200)
    revision = models.CharField(max_length=80)
    pdf = models.FileField(upload_to="manuals/")
    uploaded_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.PROTECT)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processing_status = models.CharField(max_length=20, default="pending")
    def __str__(self): return f"{self.title} — Rev {self.revision}"

class ManualChunk(models.Model):
    manual = models.ForeignKey(Manual, on_delete=models.CASCADE, related_name="chunks")
    page_number = models.PositiveIntegerField()  # 1-based PDF page number
    chunk_index = models.PositiveIntegerField(default=0)
    text = models.TextField()
    # Keep vector IDs/metadata here; the actual vectors may live in a separate vector store.
    vector_id = models.CharField(max_length=200, blank=True)
    class Meta:
        indexes = [models.Index(fields=["manual", "page_number"])]

class QuestionHistory(models.Model):
    engineer = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.PROTECT, related_name="manual_questions")
    machine = models.ForeignKey(Machine, on_delete=models.PROTECT, related_name="questions")
    question = models.TextField()
    answer = models.TextField(blank=True)
    insufficient_evidence = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

class AnswerSource(models.Model):
    history = models.ForeignKey(QuestionHistory, on_delete=models.CASCADE, related_name="sources")
    manual = models.ForeignKey(Manual, on_delete=models.PROTECT)
    page_number = models.PositiveIntegerField()
    passage = models.TextField()
