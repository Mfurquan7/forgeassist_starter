
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
        {"role_names": list(request.user.groups.values_list("name", flat=True))},
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
    queryset = Manual.objects.select_related(
        "machine", "uploaded_by"
    ).all().order_by("-uploaded_at")

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

            # Remove old chunks if the manual is being reprocessed.
            ManualChunk.objects.filter(manual=manual).delete()

            chunks = []
            for page in pages:
                for index, chunk_text in enumerate(chunk_page(page["text"])):
                    chunks.append(
                        ManualChunk(
                            manual=manual,
                            page_number=page["page_number"],
                            chunk_index=index,
                            text=chunk_text,
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

        # Save the question first so it can be revisited in history.
        history = QuestionHistory.objects.create(
            engineer=request.user,
            machine=machine,
            question=question,
        )

        try:
            # Retrieval is restricted to the selected machine.
            passages = retrieve_for_machine(machine.id, question)

            answer, insufficient = generate_grounded_answer(
                question,
                passages,
            )

            history.answer = answer
            history.insufficient_evidence = insufficient
            history.save(
                update_fields=["answer", "insufficient_evidence"]
            )

            # Save only the top-ranked source when evidence is sufficient.
            if passages and not insufficient:
                passage = passages[0]

                # Verify the source manual belongs to the selected machine.
                manual = Manual.objects.get(
                    pk=passage.manual_id,
                    machine=machine,
                )

                AnswerSource.objects.create(
                    history=history,
                    manual=manual,
                    page_number=passage.page_number,
                    passage=passage.text,
                )

        except Exception:
            # Preserve the question in history if processing fails.
            history.answer = (
                "I could not process this question. "
                "Please try again or consult an authorized maintenance engineer."
            )
            history.insufficient_evidence = True
            history.save(
                update_fields=["answer", "insufficient_evidence"]
            )

        sources = [
            {
                "manual": source.manual.title,
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
