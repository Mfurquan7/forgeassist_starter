"""Tests for domain-generic structured manual extractors."""

from django.test import TestCase

from core.services.structured_extractors import (
    extract_operating_parameters,
    extract_maintenance_items,
    extract_safety_items,
    extract_troubleshooting_items,
    extract_all_structured,
)


class StructuredExtractorsTests(TestCase):

    def test_extract_operating_parameters(self):
        text = (
            "Operating reference values\n"
            "Normal hydraulic pressure: 120 to 140 bar during forming\n"
            "Operating temperature: 40 to 60 °C\n"
            "Maximum pressure: 180 bar\n"
        )
        params = extract_operating_parameters(text)
        self.assertTrue(len(params) >= 2)
        param_names = [p["parameter"].lower() for p in params]
        self.assertTrue(any("pressure" in name for name in param_names))
        self.assertTrue(any("temperature" in name for name in param_names))

    def test_extract_maintenance_schedule(self):
        text = (
            "Maintenance Schedule:\n"
            "Daily: Check oil level, inspect hoses for leakage\n"
            "Every 250 operating hours: Replace hydraulic filter element\n"
            "Annually: Perform full fluid replacement\n"
        )
        items = extract_maintenance_items(text)
        self.assertTrue(len(items) >= 2)
        intervals = [item["interval"].lower() for item in items]
        self.assertTrue(any("daily" in i for i in intervals))

    def test_extract_safety_precautions(self):
        text = (
            "DANGER: High pressure fluid hazard.\n"
            "A guard interlock must never be bypassed.\n"
            "Wear safety glasses and protective gloves at all times.\n"
            "Always follow facility lockout/tagout procedures.\n"
        )
        precautions = extract_safety_items(text)
        self.assertTrue(len(precautions) >= 3)
        categories = [p["category"] for p in precautions]
        self.assertIn("danger", categories)
        self.assertIn("ppe", categories)
        self.assertIn("lockout", categories)

    def test_extract_troubleshooting_items(self):
        text = (
            "Problem: Low pressure\n"
            "Check: Reservoir level and RF-20 restriction indicator\n"
            "Action: Refill reservoir or replace filter\n"
        )
        items = extract_troubleshooting_items(text)
        self.assertTrue(len(items) >= 1)
        self.assertIn("Low pressure", items[0]["symptom"])

    def test_extract_all_structured(self):
        text = (
            "Normal hydraulic pressure: 120 to 140 bar\n"
            "Daily: Inspect sight glass\n"
            "WARNING: Do not open guard during cycle\n"
        )
        result = extract_all_structured(text)
        self.assertIn("operating_parameters", result)
        self.assertIn("maintenance", result)
        self.assertIn("safety", result)
        self.assertIn("troubleshooting", result)
