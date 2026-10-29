from rest_framework import serializers
from .models import Machine, Manual, QuestionHistory

class MachineSerializer(serializers.ModelSerializer):
    class Meta:
        model = Machine
        fields = ["id", "name", "model_number", "description", "created_at"]
        read_only_fields = ["id", "created_at"]

class ManualSerializer(serializers.ModelSerializer):
    machine_name = serializers.CharField(source="machine.name", read_only=True)
    class Meta:
        model = Manual
        fields = ["id", "machine", "machine_name", "title", "revision", "pdf", "uploaded_by", "uploaded_at", "processing_status"]
        read_only_fields = ["uploaded_by", "uploaded_at", "processing_status"]

class QuestionSerializer(serializers.ModelSerializer):
    class Meta:
        model = QuestionHistory
        fields = ["id", "machine", "question", "answer", "insufficient_evidence", "created_at"]
        read_only_fields = fields
