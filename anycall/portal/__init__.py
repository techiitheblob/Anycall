"""AnyCall Field Control Software & Web Portal.

Provides an edge-deployable FastAPI server and responsive browser dashboard
for real-time wildlife acoustic monitoring, few-shot species enrollment,
novel sound cluster discovery, and field configuration.
"""
from anycall.portal.api import create_app

__all__ = ["create_app"]
