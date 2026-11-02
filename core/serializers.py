
from rest_framework import serializers

from .models import (
    Machine,
    Manual,
    QuestionHistory,
    AnswerSource,
)


class MachineSerializer(serializers.ModelSerializer):
    class Meta:
        model = Machine
        fields = [
            "id",
            "name",
            "model_number",
            "description",
            "created_at",
        ]
        read_only_fields = ["id", "created_at"]


class ManualSerializer(serializers.ModelSerializer):
    machine_name = serializers.CharField(
        source="machine.name",
        read_only=True
    )

    class Meta:
        model = Manual
        fields = [
            "id",
            "machine",
            "machine_name",
            "title",
            "revision",
            "pdf",
            "uploaded_by",
            "uploaded_at",
            "processing_status",
        ]
        read_only_fields = [
            "uploaded_by",
            "uploaded_at",
            "processing_status",
        ]


class AnswerSourceSerializer(serializers.ModelSerializer):
    manual_title = serializers.CharField(
        source="manual.title",
        read_only=True
    )
    document_id = serializers.CharField(
        source="manual.document_id",
        read_only=True
    )
    revision = serializers.CharField(
        source="manual.revision",
        read_only=True
    )

    class Meta:
        model = AnswerSource
        fields = [
            "manual_title",
            "document_id",
            "revision",
            "page_number",
            "passage",
        ]


class QuestionSerializer(serializers.ModelSerializer):
    sources = AnswerSourceSerializer(
        many=True,
        read_only=True
    )

    class Meta:
        model = QuestionHistory
        fields = [
            "id",
            "machine",
            "question",
            "answer",
            "insufficient_evidence",
            "answer_status",
            "sources",
            "created_at",
        ]
        read_only_fields = fields
