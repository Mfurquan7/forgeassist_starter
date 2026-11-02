from django.contrib import admin
from django.urls import path, include
from core import views

urlpatterns = [
    path("admin/", admin.site.urls),
    path("accounts/", include("django.contrib.auth.urls")),
    path("", views.dashboard, name="dashboard"),
    path("machines/", views.machines_page, name="machines-page"),
path("manuals/", views.manuals_page, name="manuals-page"),
path("history/", views.history_page, name="history-page"),
    path("api/", include("core.urls")),
]
