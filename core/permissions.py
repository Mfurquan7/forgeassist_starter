from rest_framework.permissions import BasePermission, SAFE_METHODS

ADMIN_ROLE = "Administrator"
ENGINEER_ROLE = "Engineer"
SUPERVISOR_ROLE = "Supervisor"

def has_role(user, role):
    return user.is_authenticated and (user.is_superuser or user.groups.filter(name=role).exists())

class IsAdministrator(BasePermission):
    message = "Administrator role required."
    def has_permission(self, request, view):
        return has_role(request.user, ADMIN_ROLE)

class IsEngineer(BasePermission):
    message = "Engineer role required."
    def has_permission(self, request, view):
        return has_role(request.user, ENGINEER_ROLE)

class IsSupervisor(BasePermission):
    message = "Supervisor role required."
    def has_permission(self, request, view):
        return has_role(request.user, SUPERVISOR_ROLE)

class IsSupervisorReadOnly(BasePermission):
    message = "Supervisor role required; this view is read-only."
    def has_permission(self, request, view):
        return has_role(request.user, SUPERVISOR_ROLE) and request.method in SAFE_METHODS
