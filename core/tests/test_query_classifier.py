"""Tests for the query classification and semantic expansion service."""

from django.test import TestCase

from core.services.query_classifier import (
    classify_question,
    expand_query_terms,
    extract_error_codes,
    CATEGORY_ERROR_CODE,
    CATEGORY_OPERATING_PARAM,
    CATEGORY_MAINTENANCE,
    CATEGORY_SAFETY,
    CATEGORY_TROUBLESHOOTING,
    CATEGORY_UNSUPPORTED,
    CATEGORY_GENERAL,
)


class QueryClassifierTests(TestCase):

    def test_extract_standard_error_codes(self):
        codes = extract_error_codes("What does error E17 mean?")
        self.assertIn("E17", codes)

        codes_dash = extract_error_codes("Explain fault code E-24")
        self.assertIn("E24", codes_dash)

        codes_space = extract_error_codes("Alarm code E 08 was triggered")
        self.assertIn("E08", codes_space)

    def test_classify_error_code_question(self):
        result = classify_question("What is documented response for E31?")
        self.assertEqual(result["category"], CATEGORY_ERROR_CODE)
        self.assertIn("E31", result["error_codes"])

    def test_classify_operating_parameter_question(self):
        result = classify_question("What is the recommended operating oil temperature range?")
        self.assertEqual(result["category"], CATEGORY_OPERATING_PARAM)
        self.assertGreaterEqual(result["confidence"], 0.6)

    def test_classify_maintenance_schedule_question(self):
        result = classify_question("What are the maintenance intervals and inspection items?")
        self.assertEqual(result["category"], CATEGORY_MAINTENANCE)

    def test_classify_safety_question(self):
        result = classify_question("What safety precautions and interlock rules are documented?")
        self.assertEqual(result["category"], CATEGORY_SAFETY)

    def test_classify_troubleshooting_question(self):
        result = classify_question("What checks should be performed when hydraulic pressure is low?")
        self.assertEqual(result["category"], CATEGORY_TROUBLESHOOTING)

    def test_classify_unsupported_question(self):
        result = classify_question("What is the current weather in London?")
        self.assertEqual(result["category"], CATEGORY_UNSUPPORTED)

    def test_expand_query_terms_includes_synonyms(self):
        terms = expand_query_terms("oil temperature")
        self.assertIn("hydraulic", terms)
        self.assertIn("fluid", terms)

        maint_terms = expand_query_terms("maintenance intervals")
        self.assertIn("service", maint_terms)
        self.assertIn("schedule", maint_terms)
