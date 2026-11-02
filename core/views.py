
from django.contrib.auth.decorators import login_required
from django.shortcuts import render

from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.exceptions import ValidationError

from .models import (
    Machine,
    Manual,
    QuestionHistory,
    AnswerSource,
    ManualChunk,
)
from .permissions import (
    IsAdministrator,
    IsEngineer,
    IsSupervisorReadOnly,
    has_role,
)
from .serializers import (
    MachineSerializer,
    ManualSerializer,
    QuestionSerializer,
)
from .services.pdf_ingestion import extract_pages, chunk_page
from .services.rag import (
    retrieve_for_machine,
    generate_grounded_answer,
)


@login_required
def dashboard(request):
    return render(
        request,
        "core/dashboard.html",
        {
            "role_names": list(
                request.user.groups.values_list("name", flat=True)
            )
        },
    )


class MachineViewSet(viewsets.ModelViewSet):
    queryset = Machine.objects.all().order_by("name")
    serializer_class = MachineSerializer
    permission_classes = [IsAuthenticated]

    def get_permissions(self):
        if self.action in ["create", "update", "partial_update", "destroy"]:
            return [IsAdministrator()]

        return [IsAuthenticated()]


class ManualViewSet(viewsets.ModelViewSet):
    queryset = (
        Manual.objects.select_related("machine", "uploaded_by")
        .all()
        .order_by("-uploaded_at")
    )

    serializer_class = ManualSerializer
    parser_classes = [MultiPartParser, FormParser, JSONParser]

    def get_permissions(self):
        if self.action in ["create", "update", "partial_update", "destroy"]:
            return [IsAdministrator()]

        return [IsAuthenticated()]

    def perform_create(self, serializer):
        upload = self.request.FILES.get("pdf")

        if not upload or not upload.name.lower().endswith(".pdf"):
            raise ValidationError({"pdf": "Upload a PDF file."})

        manual = serializer.save(uploaded_by=self.request.user)

        try:
            pages = extract_pages(manual.pdf.path)

            # Remove previous chunks before reprocessing.
            ManualChunk.objects.filter(manual=manual).delete()

            chunks = []

            for page in pages:
                for index, chunk_text in enumerate(
                    chunk_page(page["text"])
                ):
                    chunks.append(
                        ManualChunk(
                            manual=manual,
                            page_number=page["page_number"],
                            chunk_index=index,
                            text=str(chunk_text),
                            chunk_type=getattr(chunk_text, "chunk_type", "text"),
                            heading_context=getattr(chunk_text, "heading_context", ""),
                        )
                    )

            if chunks:
                ManualChunk.objects.bulk_create(chunks)

            manual.processing_status = "ready" if chunks else "empty"

        except Exception:
            manual.processing_status = "failed"
            manual.save(update_fields=["processing_status"])
            raise

        manual.save(update_fields=["processing_status"])


class QuestionViewSet(viewsets.ReadOnlyModelViewSet):
    serializer_class = QuestionSerializer

    def get_queryset(self):
        queryset = (
            QuestionHistory.objects.select_related("machine", "engineer")
            .all()
            .order_by("-created_at")
        )

        if (
            has_role(self.request.user, "Supervisor")
            or self.request.user.is_superuser
        ):
            return queryset

        return queryset.filter(engineer=self.request.user)

    def get_permissions(self):
        if self.action == "create_answer":
            return [IsEngineer()]

        return [IsAuthenticated()]

    @action(detail=False, methods=["post"], url_path="ask")
    def create_answer(self, request):
        machine_id = request.data.get("machine")
        question = (request.data.get("question") or "").strip()

        if not machine_id or not question:
            return Response(
                {"detail": "machine and question are required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            machine = Machine.objects.get(pk=machine_id)

        except (Machine.DoesNotExist, ValueError, TypeError):
            return Response(
                {"detail": "Machine not found"},
                status=status.HTTP_404_NOT_FOUND,
            )

        # Save the question first for history.
        history = QuestionHistory.objects.create(
            engineer=request.user,
            machine=machine,
            question=question,
        )

        try:
            # Retrieve passages only from the selected machine.
            passages = retrieve_for_machine(
                machine.id,
                question,
            )

            result = generate_grounded_answer(
                question,
                passages,
            )

            if len(result) == 3:
                answer, insufficient, answer_status_code = result
            else:
                answer, insufficient = result
                if insufficient:
                    answer_status_code = "INSUFFICIENT_EVIDENCE"
                else:
                    answer_status_code = "ANSWERED"

            # Check if answer indicates LLM outage or review needed
            if "AI service could not generate" in answer or "service is unavailable" in answer.lower():
                answer_status_code = "LLM_UNAVAILABLE"
            elif "consult an authorized" in answer and not insufficient:
                answer_status_code = "REVIEW_REQUIRED"

            history.answer = answer
            history.insufficient_evidence = insufficient
            history.answer_status = answer_status_code

            history.save(
                update_fields=["answer", "insufficient_evidence", "answer_status"]
            )

            # Remove any previous sources if this history is reprocessed.
            history.sources.all().delete()

            # Save all retrieved sources when evidence is sufficient or when review is suggested.
            if passages and not insufficient:
                sources_to_create = []
                seen_sources = set()

                for passage in passages:
                    # Verify that the source manual belongs
                    # to the selected machine.
                    manual = Manual.objects.filter(
                        pk=passage.manual_id,
                        machine=machine,
                    ).first()

                    if not manual:
                        continue

                    # Prevent duplicate source records.
                    source_key = (
                        manual.id,
                        passage.page_number,
                        passage.text,
                    )

                    if source_key in seen_sources:
                        continue

                    seen_sources.add(source_key)

                    sources_to_create.append(
                        AnswerSource(
                            history=history,
                            manual=manual,
                            page_number=passage.page_number,
                            passage=passage.text,
                        )
                    )

                if sources_to_create:
                    AnswerSource.objects.bulk_create(
                        sources_to_create
                    )

        except Exception:
            # Preserve the question in history if processing fails.
            history.answer = (
                "I could not process this question. "
                "Please try again or consult an authorized "
                "maintenance engineer."
            )

            history.insufficient_evidence = True
            history.answer_status = "LLM_UNAVAILABLE"

            history.save(
                update_fields=["answer", "insufficient_evidence", "answer_status"]
            )

            history.sources.all().delete()

        # Return saved source details.
        sources = [
            {
                "manual": source.manual.title,
                "document_id": source.manual.document_id,
                "revision": source.manual.revision,
                "page": source.page_number,
                "passage": source.passage,
            }
            for source in history.sources.select_related("manual").all()
        ]

        return Response(
            {
                "id": history.id,
                "answer": history.answer,
                "insufficient_evidence": history.insufficient_evidence,
                "answer_status": history.answer_status,
                "sources": sources,
            },
            status=status.HTTP_200_OK,
        )


class SupervisorHistoryViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = (
        QuestionHistory.objects.select_related("machine", "engineer")
        .all()
        .order_by("-created_at")
    )

    serializer_class = QuestionSerializer
    permission_classes = [IsSupervisorReadOnly]
