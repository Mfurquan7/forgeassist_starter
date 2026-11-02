"""Unit tests for ForgeAssist AI RAG enhancements and safety guarantees."""

from unittest.mock import patch
from django.test import TestCase
from django.contrib.auth.models import User

from core.models import Machine, Manual, ManualChunk
from core.services.rag import (
    retrieve_for_machine,
    generate_grounded_answer,
    extract_direct_answer,
    INSUFFICIENT_EVIDENCE,
    REVIEW_PASSAGE_MESSAGE,
    RetrievedPassage,
)
from core.services.pdf_ingestion import chunk_page


class RAGImprovementsTests(TestCase):

    def setUp(self):
        self.user = User.objects.create_user(
            username="rag_test_user",
            password="TestPassword123!",
        )

        self.machine = Machine.objects.create(
            machine_code="TEST-HP200",
            name="Hydraulic Forming Press",
            model_number="HP-200-DEMO",
        )

        self.manual = Manual.objects.create(
            machine=self.machine,
            document_id="HP-OPS-01",
            title="HP 200 Operations Manual",
            revision="1.0",
            uploaded_by=self.user,
            processing_status="processed",
        )

        # Chunks mimicking actual manuals
        self.chunk1 = ManualChunk.objects.create(
            manual=self.manual,
            page_number=1,
            chunk_index=0,
            chunk_type="table",
            heading_context="Operating reference values",
            text=(
                "Operating reference values\n"
                "Parameter\n"
                "HP-200 reference\n"
                "Normal hydraulic pressure\n"
                "120 to 140 bar during the forming stage\n"
                "Oil temperature ready range\n"
                "35 to 55 degrees C\n"
                "Oil temperature high warning\n"
                "Above 60 degrees C\n"
                "Oil temperature high stop\n"
                "65 degrees C or higher\n"
                "Reservoir level indication\n"
                "Between MIN and MAX sight-glass marks\n"
            ),
        )

        self.chunk2 = ManualChunk.objects.create(
            manual=self.manual,
            page_number=2,
            chunk_index=1,
            chunk_type="error_table",
            heading_context="Alarm Reference",
            text=(
                "Pre-cycle review\n"
                "In the demonstration workflow, check the guard indicator. A guard interlock must never be bypassed.\n"
                "Alarm codes\n"
                "Code\n"
                "Meaning\n"
                "Documented response\n"
                "E17\n"
                "Low hydraulic pressure\n"
                "Record pressure and cycle stage; place on HOLD and request checks.\n"
                "E24\n"
                "Oil temperature high warning\n"
                "Above 60 degrees C: hold the cycle and request cooling-system inspection.\n"
            ),
        )

        self.chunk3 = ManualChunk.objects.create(
            manual=self.manual,
            page_number=3,
            chunk_index=2,
            chunk_type="text",
            heading_context="Troubleshooting",
            text=(
                "Troubleshooting Observations and Escalation\n"
                "Low hydraulic pressure or E17\n"
                "Collect these observations in order: (1) reservoir level relative to MIN and MAX; "
                "(2) visible external leakage reported by the operator; (3) RF-20 restriction-indicator state; "
                "(4) oil temperature and the recorded pressure during forming. "
                "Use the operating manual to interpret pressure. Do not adjust the relief-valve setting to compensate."
            ),
        )

    def test_section_aware_chunking_preserves_tables(self):
        sample_table = (
            "Code\nMeaning\nDocumented response\n"
            "E17\nLow pressure\nHold cycle\n"
            "E24\nHigh temperature\nRequest inspection\n"
        )
        chunks = chunk_page(sample_table)
        self.assertTrue(len(chunks) >= 1)
        self.assertEqual(chunks[0].chunk_type, "error_table")

    def test_paraphrased_temperature_retrieval(self):
        # Paraphrased query: "recommended operating oil temperature range"
        passages = retrieve_for_machine(
            self.machine.id,
            "What is the recommended operating oil temperature range?",
        )
        self.assertTrue(passages)
        self.assertEqual(passages[0].document_id, "HP-OPS-01")
        self.assertIn("35 to 55 degrees C", passages[0].text)

    def test_deterministic_oil_temperature_answer(self):
        passages = retrieve_for_machine(
            self.machine.id,
            "What is the recommended operating oil temperature range?",
        )
        answer, insufficient = generate_grounded_answer(
            "What is the recommended operating oil temperature range?",
            passages,
        )
        self.assertFalse(insufficient)
        self.assertIn("35 to 55 degrees C", answer)

    def test_deterministic_low_pressure_checks_answer(self):
        passages = retrieve_for_machine(
            self.machine.id,
            "What checks should be performed when hydraulic pressure is low?",
        )
        answer, insufficient = generate_grounded_answer(
            "What checks should be performed when hydraulic pressure is low?",
            passages,
        )
        self.assertFalse(insufficient)
        self.assertIn("reservoir level relative to MIN and MAX", answer)
        self.assertIn("visible external leakage", answer)

    def test_safety_boundary_guard_interlock(self):
        passages = retrieve_for_machine(
            self.machine.id,
            "What safety precautions are documented?",
        )
        answer, insufficient = generate_grounded_answer(
            "What safety precautions are documented?",
            passages,
        )
        self.assertFalse(insufficient)
        self.assertIn("interlock must never be bypassed", answer.lower())

    def test_llm_unavailable_fallback_uses_passage_review(self):
        passages = [
            RetrievedPassage(
                manual_id=self.manual.id,
                document_id="HP-OPS-01",
                manual_title="HP 200 Operations Manual",
                revision="1.0",
                page_number=1,
                text="Some complex narrative passage that cannot be parsed by regex.",
                score=0.9,
            )
        ]

        with patch("core.services.rag.generate_with_gemini", side_effect=Exception("API connection timeout")):
            answer, insufficient = generate_grounded_answer(
                "Can you explain the historical design background of this press?",
                passages,
            )
            self.assertTrue(insufficient)
            self.assertEqual(answer, REVIEW_PASSAGE_MESSAGE)

    def test_credential_inquiry_is_strictly_blocked(self):
        passages = retrieve_for_machine(
            self.machine.id,
            "What is the administrator password or wifi token?",
        )
        self.assertEqual(passages, [])

        answer, insufficient = generate_grounded_answer(
            "What is the administrator password or wifi token?",
            [],
        )
        self.assertTrue(insufficient)
        self.assertEqual(answer, INSUFFICIENT_EVIDENCE)
