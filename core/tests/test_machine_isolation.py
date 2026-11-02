
from django.test import TestCase
from django.contrib.auth.models import User

from core.models import Machine, Manual, ManualChunk
from core.services.rag import retrieve_for_machine


class MachineIsolationTests(TestCase):

    def setUp(self):
        self.user = User.objects.create_user(
            username="isolation_test_user",
            password="TestPassword123!"
        )

        self.hp_machine = Machine.objects.create(
            machine_code="TEST-HP",
            name="Hydraulic Press",
            model_number="TEST-HP-001"
        )

        self.ac_machine = Machine.objects.create(
            machine_code="TEST-AC",
            name="Air Compressor",
            model_number="TEST-AC-001"
        )

        self.hp_manual = Manual.objects.create(
            machine=self.hp_machine,
            document_id="TEST-HP-DOC",
            title="HP Operations Manual",
            revision="1.0",
            uploaded_by=self.user,
            processing_status="processed"
        )

        self.ac_manual = Manual.objects.create(
            machine=self.ac_machine,
            document_id="TEST-AC-DOC",
            title="AC Operations Manual",
            revision="1.0",
            uploaded_by=self.user,
            processing_status="processed"
        )

        ManualChunk.objects.create(
            manual=self.hp_manual,
            page_number=1,
            chunk_index=0,
            text="Normal hydraulic pressure is 120 to 140 bar."
        )

        ManualChunk.objects.create(
            manual=self.ac_manual,
            page_number=1,
            chunk_index=0,
            text="Normal hydraulic pressure is 200 to 220 bar."
        )

    def test_hp_retrieval_excludes_ac_manual(self):
        passages = retrieve_for_machine(
            self.hp_machine.id,
            "What is the normal hydraulic pressure range?"
        )

        self.assertTrue(passages)

        for passage in passages:
            self.assertEqual(
                passage.document_id,
                "TEST-HP-DOC"
            )

    def test_ac_retrieval_excludes_hp_manual(self):
        passages = retrieve_for_machine(
            self.ac_machine.id,
            "What is the normal hydraulic pressure range?"
        )

        self.assertTrue(passages)

        for passage in passages:
            self.assertEqual(
                passage.document_id,
                "TEST-AC-DOC"
            )

    def test_unsupported_question_returns_no_passages(self):
        passages = retrieve_for_machine(
            self.hp_machine.id,
            "What is the paint colour of the machine?"
        )

        self.assertEqual(passages, [])

    def test_relevant_question_returns_passages(self):
        passages = retrieve_for_machine(
            self.hp_machine.id,
            "What is the normal hydraulic pressure range?"
        )

        self.assertTrue(passages)

        self.assertTrue(
            all(
                passage.document_id == "TEST-HP-DOC"
                for passage in passages
            )
        )
