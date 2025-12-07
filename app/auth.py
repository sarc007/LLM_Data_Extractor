import os
from fastapi import Depends, Request, HTTPException, status, Form
from fastapi.security import HTTPBasic, HTTPBasicCredentials

AUTH_ENABLED = os.getenv("AUTH_ENABLED", "false").lower() == "true"
AUTH_USERNAME = os.getenv("AUTH_USERNAME", "admin")
AUTH_PASSWORD = os.getenv("AUTH_PASSWORD", "secret")

print(AUTH_ENABLED)
print(AUTH_USERNAME)
print(AUTH_PASSWORD)

security = HTTPBasic()


def basic_auth(credentials: HTTPBasicCredentials = Depends(security)):
    if not AUTH_ENABLED:
        return None  # no auth
    correct = (credentials.username == AUTH_USERNAME 
               and credentials.password == AUTH_PASSWORD)
    if not correct:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


async def require_login(request: Request):
    """
    Simple session-based guard for UI pages.
    If AUTH_ENABLED=false, it's a no-op.
    """
    if not AUTH_ENABLED:
        return

    user = request.session.get("user")
    if not user:
        raise HTTPException(status_code=302, headers={"Location": "/login"})


async def login_user(request: Request, username: str = Form(...), password: str = Form(...)):
    if not AUTH_ENABLED:
        # If auth disabled, just ignore
        request.session["user"] = "anonymous"
        return True

    if username == AUTH_USERNAME and password == AUTH_PASSWORD:
        request.session["user"] = username
        return True
    return False
