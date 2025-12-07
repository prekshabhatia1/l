import secrets
import os
import hmac # <-- NEW IMPORT
import hashlib # <-- Required for digest

WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "testsecret")

# The exact body string used for the calculation and in body.json
RAW_BODY_STRING = '{"message_id":"m_final","from":"+919876543210","to":"+14155550100","ts":"2025-01-15T10:00:00Z","text":"Final test message."}'

# The expected signature you've been using
EXPECTED_SIGNATURE = "750a99268f5c35276e5d268d879201a07096e94a5c9a09e075c345b1d471583d"

# The server's calculation (CORRECTED CODE)
calculated_signature = hmac.new(
    WEBHOOK_SECRET.encode('utf-8'),
    RAW_BODY_STRING.encode('utf-8'),
    hashlib.sha256 # <-- CORRECT HASH FUNCTION
).hexdigest()

print(f"Server Secret: {WEBHOOK_SECRET}")
print(f"Calculated Signature (Server): {calculated_signature}")
print(f"Expected Signature (Client): {EXPECTED_SIGNATURE}")

# The comparison used in your main.py (This is correct)
match = secrets.compare_digest(calculated_signature, EXPECTED_SIGNATURE)

print(f"\nHMAC Validation Result: {'SUCCESS' if match else 'FAILURE'}")
