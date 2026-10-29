from rest_framework.routers import DefaultRouter
from .views import MachineViewSet, ManualViewSet, QuestionViewSet, SupervisorHistoryViewSet

router = DefaultRouter()
router.register("machines", MachineViewSet, basename="machine")
router.register("manuals", ManualViewSet, basename="manual")
router.register("questions", QuestionViewSet, basename="question")
router.register("supervisor/history", SupervisorHistoryViewSet, basename="supervisor-history")
urlpatterns = router.urls
