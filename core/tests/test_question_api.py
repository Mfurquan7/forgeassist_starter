
from unittest.mock import patch

from django.test import TestCase
from django.contrib.auth import get_user_model
from django.contrib.auth.models import Group

from rest_framework.test import APIClient

from core.models import (
    Machine,
    Manual,
    QuestionHistory,
    AnswerSource,
)

from core.services.rag import RetrievedPassage


User = get_user_model()


class QuestionAPITests(TestCase):

    def setUp(self):
        self.client = APIClient()

        self.engineer = User.objects.create_user(
            username="api_engineer",
            password="TestPassword123!"
        )

        engineer_group, _ = Group.objects.get_or_create(
            name="Engineer"
        )
        self.engineer.groups.add(engineer_group)

        self.supervisor = User.objects.create_user(
            username="api_supervisor",
            password="TestPassword123!"
        )

        supervisor_group, _ = Group.objects.get_or_create(
            name="Supervisor"
        )
        self.supervisor.groups.add(supervisor_group)

        self.machine = Machine.objects.create(
            machine_code="API-HP",
            name="Hydraulic Press",
            model_number="API-HP-001"
        )

        self.manual = Manual.objects.create(
            machine=self.machine,
            document_id="API-HP-DOC",
            title="HP Operations Manual",
            revision="1.0",
            uploaded_by=self.engineer,
            processing_status="processed"
        )

    @patch("core.views.generate_grounded_answer")
    @patch("core.views.retrieve_for_machine")
    def test_engineer_can_ask_and_save_answer(
        self,
        mock_retrieve,
        mock_generate,
    ):
        self.client.force_authenticate(user=self.engineer)

        passage = RetrievedPassage(
            manual_id=self.manual.id,
            document_id="API-HP-DOC",
            manual_title="HP Operations Manual",
            revision="1.0",
            page_number=1,
            text="Normal hydraulic pressure is 120 to 140 bar.",
            score=1.0,
        )

        mock_retrieve.return_value = [passage]

        mock_generate.return_value = (
            "Normal hydraulic pressure is 120 to 140 bar.",
            False,
        )

        response = self.client.post(
            "/api/questions/ask/",
            {
                "machine": self.machine.id,
                "question": "What is the normal hydraulic pressure range?",
            },
            format="json",
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("120 to 140 bar", response.data["answer"])
        self.assertFalse(response.data["insufficient_evidence"])
        self.assertEqual(len(response.data["sources"]), 1)

        self.assertEqual(
            response.data["sources"][0]["document_id"],
            "API-HP-DOC",
        )

        self.assertEqual(
            response.data["sources"][0]["page"],
            1,
        )

        self.assertEqual(QuestionHistory.objects.count(), 1)
        self.assertEqual(AnswerSource.objects.count(), 1)

    def test_supervisor_can_view_global_history(self):
        QuestionHistory.objects.create(
            engineer=self.engineer,
            machine=self.machine,
            question="Test question",
            answer="Test answer",
        )

        self.client.force_authenticate(user=self.supervisor)

        response = self.client.get("/api/questions/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.data), 1)

        self.assertEqual(
            response.data[0]["question"],
            "Test question",
        )

    def test_supervisor_cannot_submit_question(self):
        self.client.force_authenticate(user=self.supervisor)

        response = self.client.post(
            "/api/questions/ask/",
            {
                "machine": self.machine.id,
                "question": "What is the pressure range?",
            },
            format="json",
        )

        self.assertEqual(response.status_code, 403)

    def test_supervisor_history_is_read_only(self):
        self.client.force_authenticate(user=self.supervisor)

        response = self.client.post(
            "/api/supervisor/history/",
            {},
            format="json",
        )

        self.assertEqual(response.status_code, 403)
