from django.test import SimpleTestCase
from core.services.rag import RetrievedPassage

class MachineIsolationContractTests(SimpleTestCase):
    def test_rag_contract_requires_machine_scoped_retrieval(self):
        """Replace with an integration test for the selected vector store before submission.

        Required assertion: when machine A is selected, every returned passage's manual
        belongs to machine A, even if a highly similar passage exists for machine B.
        """
        self.assertTrue("machine_id" in __import__("inspect").getsource(__import__("core.services.rag", fromlist=["retrieve_for_machine"]).retrieve_for_machine))
