"""
auth.py
-------

Authentication and account recovery for the Wellness Ledger app.

Usernames, Gmail addresses, salted password hashes, and account history
are kept in a local JSON file.

Login accepts either a username or a Gmail address.
Signup requires a valid @gmail.com address.

Account recovery:
- User requests an OTP using their registered Gmail address.
- A 6-digit OTP is emailed to them.
- OTP expires after 10 minutes.
- Maximum 5 verification attempts.
- Successful verification creates a short-lived recovery token.
- The token can be used to recover the username or reset the password.

Email credentials should be supplied through environment variables:

WL_EMAIL_ADDRESS
WL_EMAIL_APP_PASSWORD

For Gmail, WL_EMAIL_APP_PASSWORD should be a Gmail App Password,
not the normal Gmail account password.
"""

import hashlib
import json
import os
import re
import secrets
import smtplib
from pathlib import Path

from datetime import datetime, timedelta
from email.message import EmailMessage


# -------------------------------------------------------------------
# FILES / SETTINGS
# -------------------------------------------------------------------

USERS_FILE = str(Path(__file__).resolve().parent / "users.json")

OTP_EXPIRY_MINUTES = 10
OTP_MAX_ATTEMPTS = 5
RECOVERY_TOKEN_EXPIRY_MINUTES = 10

GMAIL_RE = re.compile(
    r"^[A-Za-z0-9._%+-]+@gmail\.com$",
    re.IGNORECASE,
)


# -------------------------------------------------------------------
# BASIC HELPERS
# -------------------------------------------------------------------

def is_valid_gmail(email: str) -> bool:
    return bool(GMAIL_RE.match(email.strip()))


def _load():
    if not os.path.exists(USERS_FILE):
        return {}

    with open(USERS_FILE, "r") as f:
        return json.load(f)


def _save(data):
    with open(USERS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def _hash(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode(),
        salt.encode(),
        100_000,
    ).hex()


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _now():
    return datetime.now()


# -------------------------------------------------------------------
# USER LOOKUP
# -------------------------------------------------------------------

def user_exists(username: str) -> bool:
    return username.strip().lower() in _load()


def _find_username_by_email(email: str):
    users = _load()
    email = email.strip().lower()

    for uname, rec in users.items():
        if rec.get("email", "").lower() == email:
            return uname

    return None


# -------------------------------------------------------------------
# ACCOUNT CREATION
# -------------------------------------------------------------------

def create_user(
    username: str,
    email: str,
    password: str,
) -> tuple[bool, str]:

    username = username.strip().lower()
    email = email.strip().lower()

    if not username or not email or not password:
        return False, (
            "Username, Gmail address, and password are all required."
        )

    if not is_valid_gmail(email):
        return False, (
            "Please enter a valid Gmail address "
            "(must end in @gmail.com)."
        )

    if len(password) < 8:
        return False, "Password should be at least 8 characters."

    users = _load()

    if username in users:
        return False, "That username is already taken."

    if _find_username_by_email(email):
        return False, (
            "That Gmail address is already registered — "
            "try logging in instead."
        )

    salt = secrets.token_hex(16)

    users[username] = {
        "email": email,
        "salt": salt,
        "hash": _hash(password, salt),
        "created": _now().isoformat(timespec="seconds"),
        "history": [],
    }

    _save(users)

    return True, "Account created."


# -------------------------------------------------------------------
# LOGIN
# -------------------------------------------------------------------

def verify_user(username: str, password: str) -> bool:
    users = _load()

    username = username.strip().lower()
    record = users.get(username)

    if not record:
        return False

    return _hash(password, record["salt"]) == record["hash"]


def login(
    identifier: str,
    password: str,
) -> tuple[bool, str | None]:
    """
    Log in with either a username or a Gmail address.

    Returns:
        (success, username)
    """

    identifier = identifier.strip()

    if "@" in identifier:
        username = _find_username_by_email(identifier)

        if not username:
            return False, None
    else:
        username = identifier.lower()

    if verify_user(username, password):
        return True, username

    return False, None


# -------------------------------------------------------------------
# EMAIL
# -------------------------------------------------------------------

def _send_email(
    recipient: str,
    subject: str,
    body: str,
) -> bool:

    sender = os.getenv("WL_EMAIL_ADDRESS")
    app_password = os.getenv("WL_EMAIL_APP_PASSWORD")

    if not sender or not app_password:
        print(
            "Email credentials are not configured. "
            "Set WL_EMAIL_ADDRESS and WL_EMAIL_APP_PASSWORD."
        )
        return False

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = sender
    message["To"] = recipient
    message.set_content(body)

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender, app_password)
            server.send_message(message)

        return True

    except Exception as exc:
        print(f"Email sending failed: {exc}")
        return False


# -------------------------------------------------------------------
# OTP RECOVERY
# -------------------------------------------------------------------

def request_recovery_otp(email: str) -> tuple[bool, str]:
    """
    Generate and email an OTP for account recovery.

    The OTP is stored only as a hash and expires after 10 minutes.

    Returns:
        (True, message) when the request is processed.
    """

    email = email.strip().lower()

    # Always return a generic message.
    # This avoids revealing whether an email is registered.
    generic_message = (
        "If an account exists for this Gmail address, "
        "a verification code has been sent."
    )

    if not is_valid_gmail(email):
        return True, generic_message

    username = _find_username_by_email(email)

    if not username:
        return True, generic_message

    users = _load()
    record = users[username]

    otp = f"{secrets.randbelow(1_000_000):06d}"

    otp_salt = secrets.token_hex(16)

    record["recovery"] = {
        "otp_hash": _hash(otp, otp_salt),
        "otp_salt": otp_salt,
        "otp_expires": (
            _now() + timedelta(minutes=OTP_EXPIRY_MINUTES)
        ).isoformat(),
        "otp_attempts": 0,
    }

    _save(users)

    body = f"""
Hello,

We received a request to recover your Wellness Ledger account.

Your verification code is:

{otp}

This code expires in {OTP_EXPIRY_MINUTES} minutes.

If you did not request account recovery, you can safely ignore
this email.

— Wellness Ledger
""".strip()

    sent = _send_email(
        email,
        "Wellness Ledger — Account Recovery OTP",
        body,
    )

    if not sent:
        # Remove the recovery information if the email could not
        # actually be sent.
        users = _load()

        if username in users:
            users[username].pop("recovery", None)
            _save(users)

        return False, (
            "We couldn't send the verification email right now. "
            "Please try again later."
        )

    return True, generic_message


# -------------------------------------------------------------------
# OTP VERIFICATION
# -------------------------------------------------------------------

def verify_recovery_otp(
    email: str,
    otp: str,
) -> tuple[bool, str | None]:
    """
    Verify an OTP.

    On success, creates a short-lived recovery token.

    Returns:
        (True, recovery_token)
        (False, None)
    """

    email = email.strip().lower()
    otp = otp.strip()

    username = _find_username_by_email(email)

    if not username:
        return False, None

    users = _load()
    record = users.get(username)

    if not record:
        return False, None

    recovery = record.get("recovery")

    if not recovery:
        return False, None

    # Check attempt limit.
    attempts = recovery.get("otp_attempts", 0)

    if attempts >= OTP_MAX_ATTEMPTS:
        record.pop("recovery", None)
        _save(users)
        return False, None

    # Check expiry.
    try:
        expires = datetime.fromisoformat(
            recovery["otp_expires"]
        )
    except (KeyError, ValueError):
        record.pop("recovery", None)
        _save(users)
        return False, None

    if _now() > expires:
        record.pop("recovery", None)
        _save(users)
        return False, None

    # Check OTP.
    expected_hash = recovery.get("otp_hash")
    otp_salt = recovery.get("otp_salt")

    if not expected_hash or not otp_salt:
        record.pop("recovery", None)
        _save(users)
        return False, None

    actual_hash = _hash(otp, otp_salt)

    if not secrets.compare_digest(actual_hash, expected_hash):
        recovery["otp_attempts"] = attempts + 1
        _save(users)
        return False, None

    # OTP is correct.
    recovery_token = secrets.token_urlsafe(32)

    record["recovery"] = {
        "recovery_token_hash": _hash_token(recovery_token),
        "recovery_token_expires": (
            _now()
            + timedelta(minutes=RECOVERY_TOKEN_EXPIRY_MINUTES)
        ).isoformat(),
    }

    _save(users)

    return True, recovery_token


# -------------------------------------------------------------------
# RECOVERY TOKEN VALIDATION
# -------------------------------------------------------------------

def _validate_recovery_token(
    email: str,
    recovery_token: str,
):
    email = email.strip().lower()

    username = _find_username_by_email(email)

    if not username:
        return None

    users = _load()
    record = users.get(username)

    if not record:
        return None

    recovery = record.get("recovery")

    if not recovery:
        return None

    stored_hash = recovery.get("recovery_token_hash")

    if not stored_hash:
        return None

    try:
        expires = datetime.fromisoformat(
            recovery["recovery_token_expires"]
        )
    except (KeyError, ValueError):
        return None

    if _now() > expires:
        return None

    if not secrets.compare_digest(
        _hash_token(recovery_token),
        stored_hash,
    ):
        return None

    return username


# -------------------------------------------------------------------
# RECOVER USERNAME
# -------------------------------------------------------------------

def recover_username(
    email: str,
    recovery_token: str,
) -> tuple[bool, str | None]:

    username = _validate_recovery_token(
        email,
        recovery_token,
    )

    if not username:
        return False, None

    users = _load()

    # Recovery is single-use.
    users[username].pop("recovery", None)

    _save(users)

    return True, username


# -------------------------------------------------------------------
# RESET PASSWORD
# -------------------------------------------------------------------

def reset_password(
    email: str,
    recovery_token: str,
    new_password: str,
) -> tuple[bool, str]:

    if len(new_password) < 8:
        return False, "Password should be at least 8 characters."

    username = _validate_recovery_token(
        email,
        recovery_token,
    )

    if not username:
        return False, (
            "Your recovery session has expired. "
            "Please request a new OTP."
        )

    users = _load()

    salt = secrets.token_hex(16)

    users[username]["salt"] = salt
    users[username]["hash"] = _hash(
        new_password,
        salt,
    )

    # Make recovery token single-use.
    users[username].pop("recovery", None)

    _save(users)

    return True, "Password reset successfully."


# -------------------------------------------------------------------
# HISTORY
# -------------------------------------------------------------------

def add_history_entry(
    username: str,
    score: float,
    top_factor_label: str,
):
    users = _load()

    username = username.strip().lower()

    if username not in users:
        return

    users[username].setdefault(
        "history",
        [],
    ).append(
        {
            "date": _now().isoformat(timespec="seconds"),
            "score": round(score, 1),
            "top_factor": top_factor_label,
        }
    )

    _save(users)


def get_history(username: str):
    users = _load()

    username = username.strip().lower()

    return users.get(
        username,
        {},
    ).get(
        "history",
        [],
    )