from django.contrib.auth import get_user_model
from django.contrib.auth.models import Group
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from rest_framework.test import APIClient
from core.models import Machine

User = get_user_model()

class RolePermissionTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.engineer = User.objects.create_user(username="engineer", password="test-pass-123")
        group, _ = Group.objects.get_or_create(name="Engineer")
        self.engineer.groups.add(group)
        self.machine = Machine.objects.create(name="Machine A", model_number="A-100")

    def test_engineer_cannot_upload_manual(self):
        self.client.force_authenticate(self.engineer)
        fake_pdf = SimpleUploadedFile("manual.pdf", b"not-a-real-pdf", content_type="application/pdf")
        response = self.client.post("/api/manuals/", {"machine": self.machine.id, "title": "Test", "revision": "A", "pdf": fake_pdf}, format="multipart")
        self.assertEqual(response.status_code, 403)

    def test_engineer_cannot_create_machine(self):
        self.client.force_authenticate(self.engineer)
        response = self.client.post("/api/machines/", {"name": "Unauthorized", "model_number": "NO-1"}, format="json")
        self.assertEqual(response.status_code, 403)
