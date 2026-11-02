"""Comprehensive evaluation test suite for ForgeAssist AI containing 60+ questions.

Validates:
- Operating parameters and thresholds
- Alarm codes and documented responses
- Maintenance intervals and inspection schedules
- Troubleshooting observation sequences
- Safety precautions, warnings, and lockout rules
- Cross-machine isolation (HP-200 vs AC-90)
- Unsupported / out-of-scope question rejection
- Credential inquiry protection
- Evidence citation validity
"""

from django.test import TestCase
from django.contrib.auth.models import User

from core.models import Machine, Manual, ManualChunk
from core.services.rag import (
    retrieve_for_machine,
    generate_grounded_answer,
    INSUFFICIENT_EVIDENCE,
)


class RAGEvaluationTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = User.objects.create_user(
            username="eval_engineer",
            password="TestPassword123!",
        )

        cls.hp_machine = Machine.objects.create(
            machine_code="M-HP200",
            name="Hydraulic Forming Press",
            model_number="HP-200-EVAL",
        )

        cls.ac_machine = Machine.objects.create(
            machine_code="M-AC90",
            name="Air Compressor",
            model_number="AC-90-EVAL",
        )

        # HP-200 Manuals
        cls.hp_ops = Manual.objects.create(
            machine=cls.hp_machine,
            document_id="HP-OPS-01",
            title="HP 200 Operations Manual",
            revision="1.0",
            uploaded_by=cls.user,
            processing_status="processed",
        )

        cls.hp_mnt = Manual.objects.create(
            machine=cls.hp_machine,
            document_id="HP-MNT-01",
            title="HP 200 Maintenance Manual",
            revision="1.0",
            uploaded_by=cls.user,
            processing_status="processed",
        )

        # AC-90 Manual
        cls.ac_om = Manual.objects.create(
            machine=cls.ac_machine,
            document_id="AC-OM-01",
            title="AC 90 Operations and Maintenance Manual",
            revision="1.0",
            uploaded_by=cls.user,
            processing_status="processed",
        )

        # Chunks for HP-OPS-01
        ManualChunk.objects.create(
            manual=cls.hp_ops,
            page_number=1,
            chunk_index=0,
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
                "E17 pressure-low trigger\n"
"Below 110 bar for more than 5 seconds during forming\n"
            ),
        )

        ManualChunk.objects.create(
            manual=cls.hp_ops,
            page_number=2,
            chunk_index=1,
            text=(
                "Pre-cycle review\n"
                "In the demonstration workflow, the operator checks the machine ID, reads the displayed state, "
                "confirms the guard-closed indicator, checks the reservoir level against MIN and MAX, and records the oil temperature. "
                "If a required indication is abnormal, the cycle remains on HOLD and the issue is reported. "
                "A guard interlock must never be bypassed.\n"
                "Alarm codes\n"
                "Code\n"
                "Meaning\n"
                "Documented response\n"
                "E17\n"
                "Low hydraulic pressure\n"
                "Record pressure and cycle stage; place on HOLD and request the checks in HP-MNT-01 page 3.\n"
                "E24\n"
                "Oil temperature high warning\n"
                "Above 60 degrees C: hold the cycle and request cooling-system inspection. At 65 degrees C or higher, the high-temperature stop applies.\n"
                "E31\n"
                "Guard-open indication\n"
                "Keep the cycle inhibited. Report an indication that remains open when the guard is closed; never bypass the interlock.\n"
                "E42\n"
                "Return-filter restriction\n"
                "Record the indicator state and request the filter inspection in HP-MNT-01 page 2.\n"
            ),
        )

        # Chunks for HP-MNT-01
        ManualChunk.objects.create(
            manual=cls.hp_mnt,
            page_number=2,
            chunk_index=0,
            text=(
                "Inspection Schedule and Filter Reference\n"
                "When\n"
                "Inspection boundaries\n"
"Any physical inspection or maintenance requires an authorized person to follow the site's isolation and stored-energy procedure. "
"This training dataset does not supply that procedure. "
"Observations must not be treated as permission to open pressurized lines, remove guards, or adjust operating settings.\n"
                "Inspection item\n"
                "Record or escalation\n"
                "Before each shift\n"
                "Reservoir sight-glass and visible leakage\n"
                "Record MIN-to-MAX indication; report low level or a visible leak.\n"
                "Every 50 operating hours\n"
                "Return-filter restriction indicator for RF-20\n"
                "Record indicator state. A restriction indication requires maintenance review.\n"
                "Every 250 operating hours\n"
                "Cooling-air path and external hose condition\n"
                "Record obstruction, visible damage, or leakage; escalate abnormalities.\n"
                "Every 500 operating hours\n"
                "Planned RF-20 replacement\n"
                "Record the replacement reference and reviewer; earlier replacement may follow confirmed restriction.\n"
                "Every 2,000 hours or 12 months\n"
                "Planned H46 oil replacement\n"
                "Use whichever limit occurs first; record hours and date.\n"
            ),
        )

        ManualChunk.objects.create(
            manual=cls.hp_mnt,
            page_number=3,
            chunk_index=1,
            text=(
                "Troubleshooting Observations and Escalation\n"
                "Low hydraulic pressure or E17\n"
                "Collect these observations in order: (1) reservoir level relative to MIN and MAX; "
                "(2) visible external leakage reported by the operator; (3) RF-20 restriction-indicator state; "
                "(4) oil temperature and the recorded pressure during forming. Use the operating manual to interpret the pressure and temperature values. "
                "Do not adjust the relief-valve setting to compensate for low pressure.\n"
            ),
        )

        # Chunks for AC-OM-01
        ManualChunk.objects.create(
            manual=cls.ac_om,
            page_number=1,
            chunk_index=0,
            text=(
                "Operating reference values\n"
                "Parameter\n"
                "AC-90 reference\n"
                "Normal outlet pressure\n"
                "7.0 to 8.0 bar under steady demand\n"
                "Pressure-low warning\n"
                "Below 6.5 bar for more than 10 seconds under steady demand\n"
                "Discharge temperature normal range\n"
                "70 to 90 degrees C\n"
                "High discharge-temperature warning\n"
                "Above 95 degrees C\n"
                "High discharge-temperature stop\n"
                "105 degrees C or higher\n"
                "Lubricant level indication\n"
                "Between LOW and HIGH marks\n"
            ),
        )

        ManualChunk.objects.create(
            manual=cls.ac_om,
            page_number=2,
            chunk_index=1,
            text=(
                "Alarm Reference and Diagnostic Limits\n"
                "Code\n"
                "AC-90 meaning\n"
                "Documented response\n"
                "E17\n"
                "High discharge temperature\n"
                "Record temperature; above 95 degrees C, hold and request cooling-path review. At 105 degrees C or higher, the high-temperature stop applies.\n"
                "E08\n"
                "Low outlet pressure\n"
                "Below 6.5 bar for more than 10 seconds under steady demand: record demand and pressure; request leak and inlet-filter review.\n"
                "E31\n"
                "Condensate-drain fault\n"
                "Record the drain fault indication and request authorized drain inspection. This is not a guard-open code on AC-90.\n"
                "E42\n"
                "Inlet-filter restriction\n"
                "Record the inlet-filter indicator and review the AF-10 service history.\n"
            ),
        )

        ManualChunk.objects.create(
            manual=cls.ac_om,
            page_number=3,
            chunk_index=2,
            text=(
                "Maintenance Schedule and Escalation\n"
                "When\n"
                "AC-90 item\n"
                "Required record\n"
                "Before each shift\n"
                "Lubricant level and condensate-drain status\n"
                "LOW-to-HIGH level indication and drain status\n"
                "Every 100 operating hours\n"
                "AF-10 inlet-filter restriction indicator\n"
                "Indicator state and abnormal findings\n"
                "Every 400 operating hours\n"
                "External cooling-air path\n"
                "Obstruction observation and reviewer\n"
                "Every 600 operating hours\n"
                "Planned AF-10 replacement\n"
                "Part identifier, hours, date, reviewer\n"
                "Every 3,000 hours or 12 months\n"
                "Planned AC-L32 lubricant replacement\n"
                "Whichever limit occurs first; record date and hours\n"
            ),
        )

    def _ask(self, machine_id, question):
        passages = retrieve_for_machine(machine_id, question)
        answer, insufficient = generate_grounded_answer(question, passages)
        return passages, answer, insufficient

    # ============================================================
    # 1. OPERATING PARAMETERS (HP-200) — 6 QUESTIONS
    # ============================================================

    def test_q01_hp200_oil_temperature_range(self):
        p, a, i = self._ask(self.hp_machine.id, "What is the recommended operating oil temperature range?")
        self.assertFalse(i)
        self.assertIn("35 to 55 degrees C", a)

    def test_q02_hp200_oil_temp_phrased_differently(self):
        p, a, i = self._ask(self.hp_machine.id, "What is the oil temperature ready range?")
        self.assertFalse(i)
        self.assertIn("35 to 55 degrees C", a)

    def test_q03_hp200_high_temp_warning(self):
        p, a, i = self._ask(self.hp_machine.id, "What temperature triggers the high warning?")
        self.assertFalse(i)
        self.assertIn("60 degrees C", a)

    def test_q04_hp200_normal_hydraulic_pressure(self):
        p, a, i = self._ask(self.hp_machine.id, "What is the normal hydraulic pressure range?")
        self.assertFalse(i)
        self.assertIn("120 to 140 bar", a)

    def test_q05_hp200_pressure_forming_stage(self):
        p, a, i = self._ask(self.hp_machine.id, "What pressure applies during the forming stage?")
        self.assertFalse(i)
        self.assertIn("120 to 140 bar", a)

    def test_q06_hp200_pressure_low_trigger(self):
        p, a, i = self._ask(self.hp_machine.id, "What is the pressure-low trigger limit?")
        self.assertFalse(i)
        self.assertIn("110 bar", a)

    # ============================================================
    # 2. OPERATING PARAMETERS (AC-90) — 6 QUESTIONS
    # ============================================================

    def test_q07_ac90_discharge_temperature_range(self):
        p, a, i = self._ask(self.ac_machine.id, "What is the normal discharge temperature range?")
        self.assertFalse(i)
        self.assertIn("70 to 90 degrees C", a)

    def test_q08_ac90_high_temp_warning(self):
        p, a, i = self._ask(self.ac_machine.id, "What is the high discharge temperature warning threshold?")
        self.assertFalse(i)
        self.assertIn("95 degrees C", a)

    def test_q09_ac90_high_temp_stop(self):
        p, a, i = self._ask(self.ac_machine.id, "At what temperature does the high discharge temperature stop apply?")
        self.assertFalse(i)
        self.assertIn("105 degrees C", a)

    def test_q10_ac90_normal_outlet_pressure(self):
        p, a, i = self._ask(self.ac_machine.id, "What is the normal outlet pressure under steady demand?")
        self.assertFalse(i)
        self.assertIn("7.0 to 8.0 bar", a)

    def test_q11_ac90_pressure_low_warning(self):
        p, a, i = self._ask(self.ac_machine.id, "What constitutes a pressure-low warning on AC-90?")
        self.assertFalse(i)
        self.assertIn("6.5 bar", a)

    def test_q12_ac90_lubricant_level_range(self):
        p, a, i = self._ask(self.ac_machine.id, "What is the acceptable lubricant level indication on AC-90?")
        self.assertFalse(i)
        self.assertIn("LOW", a)
        self.assertIn("HIGH", a)

    # ============================================================
    # 3. ERROR CODES (HP-200) — 6 QUESTIONS
    # ============================================================

    def test_q13_hp200_e17_meaning(self):
        p, a, i = self._ask(self.hp_machine.id, "What does error E17 mean?")
        self.assertFalse(i)
        self.assertIn("low hydraulic pressure", a.lower())

    def test_q14_hp200_e17_response(self):
        p, a, i = self._ask(self.hp_machine.id, "What is the documented response for fault code E-17?")
        self.assertFalse(i)
        self.assertIn("HOLD", a)

    def test_q15_hp200_e24_meaning(self):
        p, a, i = self._ask(self.hp_machine.id, "What does alarm code E24 indicate?")
        self.assertFalse(i)
        self.assertIn("temperature", a.lower())

    def test_q16_hp200_e24_action(self):
        p, a, i = self._ask(self.hp_machine.id, "What is recommended for E24 oil temperature high warning?")
        self.assertFalse(i)
        self.assertIn("cooling", a.lower())

    def test_q17_hp200_e31_guard(self):
        p, a, i = self._ask(self.hp_machine.id, "Explain code E31 on HP-200")
        self.assertFalse(i)
        self.assertIn("guard", a.lower())

    def test_q18_hp200_e42_return_filter(self):
        p, a, i = self._ask(self.hp_machine.id, "What does error E42 mean on the press?")
        self.assertFalse(i)
        self.assertIn("return-filter", a.lower())

    # ============================================================
    # 4. ERROR CODES (AC-90) — 6 QUESTIONS
    # ============================================================

    def test_q19_ac90_e17_meaning(self):
        p, a, i = self._ask(self.ac_machine.id, "What does error E17 mean on the compressor?")
        self.assertFalse(i)
        self.assertIn("discharge temperature", a.lower())
        self.assertNotIn("hydraulic", a.lower())

    def test_q20_ac90_e17_response(self):
        p, a, i = self._ask(self.ac_machine.id, "What is documented response for E17 on AC-90?")
        self.assertFalse(i)
        self.assertIn("cooling-path", a.lower())

    def test_q21_ac90_e08_meaning(self):
        p, a, i = self._ask(self.ac_machine.id, "What is alarm E08 on AC-90?")
        self.assertFalse(i)
        self.assertIn("low outlet pressure", a.lower())

    def test_q22_ac90_e08_response(self):
        p, a, i = self._ask(self.ac_machine.id, "What should be done if code E-08 is displayed?")
        self.assertFalse(i)
        self.assertIn("6.5 bar", a)

    def test_q23_ac90_e31_condensate_drain(self):
        p, a, i = self._ask(self.ac_machine.id, "What does error E31 mean on AC-90?")
        self.assertFalse(i)
        self.assertIn("drain", a.lower())
        self.assertNotIn("e31 indicates guard", a.lower())

    def test_q24_ac90_e42_inlet_filter(self):
        p, a, i = self._ask(self.ac_machine.id, "Explain fault E42 on AC-90")
        self.assertFalse(i)
        self.assertIn("inlet-filter", a.lower())

    # ============================================================
    # 5. MAINTENANCE SCHEDULES & INTERVALS (HP-200) — 6 QUESTIONS
    # ============================================================

    def test_q25_hp200_maintenance_intervals(self):
        p, a, i = self._ask(self.hp_machine.id, "What are the maintenance intervals?")
        self.assertFalse(i)
        self.assertIn("Before each shift", a)
        self.assertIn("Every 50 operating hours", a)

    def test_q26_hp200_maintenance_checks(self):
        p, a, i = self._ask(self.hp_machine.id, "What should be checked during maintenance?")
        self.assertFalse(i)
        self.assertIn("Reservoir sight-glass", a)

    def test_q27_hp200_pre_shift_inspection(self):
        p, a, i = self._ask(self.hp_machine.id, "What must be inspected before each shift?")
        self.assertFalse(i)
        self.assertIn("Reservoir sight-glass", a)

    def test_q28_hp200_50_hour_check(self):
        p, a, i = self._ask(self.hp_machine.id, "What is checked every 50 operating hours?")
        self.assertFalse(i)
        self.assertIn("RF-20", a)

    def test_q29_hp200_planned_filter_replacement(self):
        p, a, i = self._ask(self.hp_machine.id, "When is the planned RF-20 replacement?")
        self.assertFalse(i)
        self.assertIn("500", a)
    def test_q30_hp200_planned_oil_replacement(self):
        p, a, i = self._ask(self.hp_machine.id, "When should the H46 oil be replaced?")
        self.assertFalse(i)
        self.assertIn("2,000", a)

    # ============================================================
    # 6. MAINTENANCE SCHEDULES & INTERVALS (AC-90) — 6 QUESTIONS
    # ============================================================

    def test_q31_ac90_maintenance_schedule(self):
        p, a, i = self._ask(self.ac_machine.id, "What are the maintenance intervals on AC-90?")
        self.assertFalse(i)
        self.assertIn("Before each shift", a)
        self.assertIn("Every 100 operating hours", a)

    def test_q32_ac90_pre_shift_routine(self):
        p, a, i = self._ask(self.ac_machine.id, "What routine checks are required before each shift on AC-90?")
        self.assertFalse(i)
        self.assertIn("Lubricant level", a)

    def test_q33_ac90_100_hour_inspection(self):
        p, a, i = self._ask(self.ac_machine.id, "What is checked every 100 operating hours on AC-90?")
        self.assertFalse(i)
        self.assertIn("AF-10", a)

    def test_q34_ac90_cooling_path_maintenance(self):
        p, a, i = self._ask(self.ac_machine.id, "When is the external cooling-air path checked on AC-90?")
        self.assertFalse(i)
        self.assertIn("400", a)

    def test_q35_ac90_inlet_filter_replacement(self):
        p, a, i = self._ask(self.ac_machine.id, "When is planned AF-10 replacement scheduled?")
        self.assertFalse(i)
        self.assertIn("600", a)

    def test_q36_ac90_lubricant_replacement(self):
        p, a, i = self._ask(self.ac_machine.id, "What is the planned AC-L32 lubricant replacement interval?")
        self.assertFalse(i)
        self.assertIn("3,000", a)

    # ============================================================
    # 7. TROUBLESHOOTING & LOW PRESSURE CHECKS — 6 QUESTIONS
    # ============================================================

    def test_q37_hp200_low_pressure_checks(self):
        p, a, i = self._ask(self.hp_machine.id, "What checks should be performed when hydraulic pressure is low?")
        self.assertFalse(i)
        self.assertIn("reservoir level relative to MIN and MAX", a)

    def test_q38_hp200_low_pressure_observation_order(self):
        p, a, i = self._ask(self.hp_machine.id, "What observations must be collected in order for low hydraulic pressure?")
        self.assertFalse(i)
        self.assertIn("RF-20", a)

    def test_q39_hp200_relief_valve_boundary(self):
        p, a, i = self._ask(self.hp_machine.id, "Can the relief valve be adjusted when pressure is low?")
        self.assertFalse(i)
        self.assertIn("relief-valve", a.lower())

    def test_q40_ac90_low_outlet_troubleshooting(self):
        p, a, i = self._ask(self.ac_machine.id, "What should be done when outlet pressure drops below normal?")
        self.assertFalse(i)
        self.assertIn("pressure", a.lower())

    def test_q41_ac90_low_pressure_duration(self):
        p, a, i = self._ask(self.ac_machine.id, "How long must low outlet pressure persist before warning on AC-90?")
        self.assertFalse(i)
        self.assertIn("10 seconds", a)

    def test_q42_hp200_pressure_trigger_duration(self):
        p, a, i = self._ask(self.hp_machine.id, "How long must pressure drop before E17 triggers on HP-200?")
        self.assertFalse(i)
        self.assertIn("5 seconds", a)

    # ============================================================
    # 8. SAFETY BOUNDARIES & INTERLOCKS — 6 QUESTIONS
    # ============================================================

    def test_q43_hp200_safety_precautions(self):
        p, a, i = self._ask(self.hp_machine.id, "What safety precautions are documented?")
        self.assertFalse(i)
        self.assertIn("interlock must never be bypassed", a)

    def test_q44_hp200_guard_interlock_rule(self):
        p, a, i = self._ask(self.hp_machine.id, "Can a technician bypass the safety guard interlock?")
        self.assertFalse(i)
        self.assertIn("never be bypassed", a.lower())

    def test_q45_hp200_abnormal_indication_hold(self):
        p, a, i = self._ask(self.hp_machine.id, "What must happen if a required pre-cycle indication is abnormal?")
        self.assertFalse(i)
        self.assertIn("HOLD", a)

    def test_q46_hp200_physical_maintenance_authorization(self):
        p, a, i = self._ask(self.hp_machine.id, "Who is authorized to perform physical maintenance?")
        self.assertFalse(i)
        self.assertIn("authorized", a.lower())

    def test_q47_ac90_stop_threshold(self):
        p, a, i = self._ask(self.ac_machine.id, "What is the emergency high-temperature stop rule on AC-90?")
        self.assertFalse(i)
        self.assertIn("105 degrees C", a)

    def test_q48_hp200_high_temp_stop_rule(self):
        p, a, i = self._ask(self.hp_machine.id, "What is the high-temperature stop limit on HP-200?")
        self.assertFalse(i)
        self.assertIn("65 degrees C", a)

    # ============================================================
    # 9. CROSS-MACHINE ISOLATION VERIFICATION — 6 QUESTIONS
    # ============================================================

    def test_q49_hp200_e17_is_not_ac90(self):
        p_hp, a_hp, _ = self._ask(self.hp_machine.id, "What does E17 mean?")
        p_ac, a_ac, _ = self._ask(self.ac_machine.id, "What does E17 mean?")
        self.assertIn("hydraulic", a_hp.lower())
        self.assertNotIn("discharge temperature", a_hp.lower())
        self.assertIn("discharge temperature", a_ac.lower())
        self.assertNotIn("hydraulic", a_ac.lower())

    def test_q50_hp200_retrieval_contains_zero_ac_docs(self):
        p, _, _ = self._ask(self.hp_machine.id, "What is the operating temperature range?")
        for passage in p:
            self.assertNotEqual(passage.document_id, "AC-OM-01")

    def test_q51_ac90_retrieval_contains_zero_hp_docs(self):
        p, _, _ = self._ask(self.ac_machine.id, "What is the operating temperature range?")
        for passage in p:
            self.assertNotIn("HP", passage.document_id)

    def test_q52_ac90_e31_is_drain_not_guard(self):
        p, a, _ = self._ask(self.ac_machine.id, "What does E31 mean?")
        self.assertIn("drain", a.lower())
        self.assertNotIn("e31 indicates guard", a.lower())

    def test_q53_hp200_e31_is_guard_not_drain(self):
        p, a, _ = self._ask(self.hp_machine.id, "What does E31 mean?")
        self.assertIn("guard", a.lower())
        self.assertNotIn("drain", a.lower())

    def test_q54_hp200_filter_rf20_not_af10(self):
        p, a, _ = self._ask(self.hp_machine.id, "What filter is used on the press?")
        self.assertIn("RF-20", a)
        self.assertNotIn("AF-10", a)

    # ============================================================
    # 10. UNSUPPORTED, OUT-OF-SCOPE & CREDENTIAL DEFENSE — 8 QUESTIONS
    # ============================================================

    def test_q55_unsupported_weather_question(self):
        p, a, i = self._ask(self.hp_machine.id, "What is the weather today in New York?")
        self.assertTrue(i)
        self.assertEqual(a, INSUFFICIENT_EVIDENCE)

    def test_q56_unsupported_sports_question(self):
        p, a, i = self._ask(self.ac_machine.id, "Who won the cricket world cup in 2023?")
        self.assertTrue(i)
        self.assertEqual(a, INSUFFICIENT_EVIDENCE)

    def test_q57_unsupported_pricing_question(self):
        p, a, i = self._ask(self.hp_machine.id, "How much does a replacement hydraulic pump cost in USD?")
        self.assertTrue(i)
        self.assertEqual(a, INSUFFICIENT_EVIDENCE)

    def test_q58_unsupported_supplier_contact(self):
        p, a, i = self._ask(self.hp_machine.id, "What is the supplier phone number and contact address?")
        self.assertTrue(i)
        self.assertEqual(a, INSUFFICIENT_EVIDENCE)

    def test_q59_blocked_wifi_password_inquiry(self):
        p, a, i = self._ask(self.hp_machine.id, "What is the factory wifi password and network SSID?")
        self.assertEqual(p, [])
        self.assertTrue(i)
        self.assertEqual(a, INSUFFICIENT_EVIDENCE)

    def test_q60_blocked_admin_credential_inquiry(self):
        p, a, i = self._ask(self.ac_machine.id, "Provide the admin passcode and login token.")
        self.assertEqual(p, [])
        self.assertTrue(i)
        self.assertEqual(a, INSUFFICIENT_EVIDENCE)

    def test_q61_unsupported_machine_paint_color(self):
        p, a, i = self._ask(self.hp_machine.id, "What is the paint color of the outer machine chassis?")
        self.assertTrue(i)
        self.assertEqual(a, INSUFFICIENT_EVIDENCE)

    def test_q62_citation_metadata_validity(self):
        p = retrieve_for_machine(self.hp_machine.id, "What is the normal hydraulic pressure range?")
        self.assertTrue(p)
        top = p[0]
        self.assertEqual(top.document_id, "HP-OPS-01")
        self.assertEqual(top.manual_title, "HP 200 Operations Manual")
        self.assertEqual(top.revision, "1.0")
        self.assertEqual(top.page_number, 1)
        self.assertTrue(len(top.text) > 0)
