
from django.contrib import admin, messages
from .models import Machine, Manual, ManualChunk, QuestionHistory, AnswerSource


@admin.register(Machine)
class MachineAdmin(admin.ModelAdmin):
    list_display = ("name", "model_number", "created_at")
    search_fields = ("name", "model_number")


@admin.register(Manual)
class ManualAdmin(admin.ModelAdmin):
    list_display = (
        "title",
        "machine",
        "revision",
        "processing_status",
        "uploaded_at",
    )
    list_filter = ("machine", "processing_status")

    def save_model(self, request, obj, form, change):
        from core.services.pdf_ingestion import process_manual

        super().save_model(request, obj, form, change)

        try:
            count = process_manual(obj)
            self.message_user(
                request,
                f"PDF processed successfully: {count} chunks created.",
                messages.SUCCESS,
            )
        except Exception as exc:
            self.message_user(
                request,
                f"PDF processing failed: {exc}",
                messages.ERROR,
            )


admin.site.register(ManualChunk)
admin.site.register(QuestionHistory)
admin.site.register(AnswerSource)
