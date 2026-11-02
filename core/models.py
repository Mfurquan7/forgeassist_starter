
from django.conf import settings
from django.db import models


class Machine(models.Model):
    machine_code = models.CharField(
        max_length=50,
        unique=True,
        null=True,
        blank=True
    )
    name = models.CharField(max_length=120)
    model_number = models.CharField(max_length=120, unique=True)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.name} ({self.model_number})"


class Manual(models.Model):
    machine = models.ForeignKey(
        Machine,
        on_delete=models.CASCADE,
        related_name="manuals"
    )
    document_id = models.CharField(
        max_length=80,
        unique=True,
        null=True,
        blank=True
    )
    title = models.CharField(max_length=200)
    revision = models.CharField(max_length=80)
    pdf = models.FileField(upload_to="manuals/")
    uploaded_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.PROTECT
    )
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processing_status = models.CharField(max_length=20, default="pending")

    def __str__(self):
        return f"{self.title} - Rev {self.revision}"


class ManualChunk(models.Model):
    CHUNK_TYPE_CHOICES = [
        ("text", "Text"),
        ("table", "Table"),
        ("error_table", "Error Table"),
        ("heading", "Heading"),
        ("section", "Section"),
    ]

    manual = models.ForeignKey(
        Manual,
        on_delete=models.CASCADE,
        related_name="chunks"
    )
    page_number = models.PositiveIntegerField()
    chunk_index = models.PositiveIntegerField(default=0)
    text = models.TextField()
    vector_id = models.CharField(max_length=200, blank=True)
    chunk_type = models.CharField(
        max_length=20,
        choices=CHUNK_TYPE_CHOICES,
        default="text",
    )
    heading_context = models.CharField(
        max_length=500,
        blank=True,
        default="",
        help_text="Parent heading(s) for this chunk, providing context.",
    )

    class Meta:
        indexes = [
            models.Index(fields=["manual", "page_number"]),
            models.Index(fields=["chunk_type"]),
        ]


class QuestionHistory(models.Model):
    ANSWER_STATUS_CHOICES = [
        ("ANSWERED", "Answered"),
        ("INSUFFICIENT_EVIDENCE", "Insufficient Evidence"),
        ("LLM_UNAVAILABLE", "LLM Unavailable"),
        ("REVIEW_REQUIRED", "Review Required"),
    ]

    engineer = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.PROTECT,
        related_name="manual_questions"
    )
    machine = models.ForeignKey(
        Machine,
        on_delete=models.PROTECT,
        related_name="questions"
    )
    question = models.TextField()
    answer = models.TextField(blank=True)
    insufficient_evidence = models.BooleanField(default=False)
    answer_status = models.CharField(
        max_length=25,
        choices=ANSWER_STATUS_CHOICES,
        default="ANSWERED",
    )
    created_at = models.DateTimeField(auto_now_add=True)


class AnswerSource(models.Model):
    history = models.ForeignKey(
        QuestionHistory,
        on_delete=models.CASCADE,
        related_name="sources"
    )
    manual = models.ForeignKey(
        Manual,
        on_delete=models.PROTECT
    )
    page_number = models.PositiveIntegerField()
    passage = models.TextField()
