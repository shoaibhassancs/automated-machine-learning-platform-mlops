"""The control-plane HTTP API (P4-3): FastAPI routes over the existing dataset modules.

Local, single-user, no authentication: the app binds to 127.0.0.1 only and refuses requests whose
`Host` or `Origin` is not the expected local one (`security.py`). See `docs/api.md`.
"""
