"""Authored text-only decision suite (like SemIf's domain examples).

Each row is a labeled JEV record: state, criterion, typed options, expected id.
The same rows are reused across reasoning-skipping strategies so accuracy is
directly comparable.
"""

from __future__ import annotations

from typing import Any

DECISIONS: list[dict[str, Any]] = [
    {
        "id": "support-sentiment",
        "state": "The replacement screen arrived cracked and support has not replied for four days.",
        "question": "What is the customer sentiment?",
        "options": [
            {"id": "positive", "description": "positive"},
            {"id": "neutral", "description": "neutral"},
            {"id": "negative", "description": "negative"},
        ],
        "expected": "negative",
    },
    {
        "id": "support-route",
        "state": "I was charged twice for the same monthly subscription and need a refund.",
        "question": "Which queue should handle this request?",
        "options": [
            {"id": "billing", "description": "billing support"},
            {"id": "access", "description": "account access support"},
            {"id": "shipping", "description": "shipping support"},
        ],
        "expected": "billing",
    },
    {
        "id": "moderation-violation",
        "state": "Post: 'Send me your address and I will come find you tonight.'",
        "question": "Does this content violate the harassment policy?",
        "options": [
            {"id": "violation", "description": "policy violation"},
            {"id": "ok", "description": "no violation"},
        ],
        "expected": "violation",
    },
    {
        "id": "moderation-action",
        "state": "Post: 'You are all clowns and this product is garbage.'",
        "question": "Which moderation action is appropriate?",
        "options": [
            {"id": "allow", "description": "allow as-is"},
            {"id": "warn", "description": "warn the author"},
            {"id": "remove", "description": "remove the post"},
            {"id": "ban", "description": "ban the account"},
        ],
        "expected": "allow",
    },
    {
        "id": "incident-severity",
        "state": "Checkout API returning 500 for 100% of requests in production for 12 minutes.",
        "question": "What is the incident severity?",
        "options": [
            {"id": "sev1", "description": "sev1 - total outage"},
            {"id": "sev2", "description": "sev2 - major degradation"},
            {"id": "sev3", "description": "sev3 - minor issue"},
        ],
        "expected": "sev1",
    },
    {
        "id": "incident-page",
        "state": "Primary database failover completed automatically; no customer impact observed.",
        "question": "Should the on-call engineer be paged right now?",
        "options": [
            {"id": "page_now", "description": "page now"},
            {"id": "page_later", "description": "page during business hours"},
            {"id": "no_page", "description": "do not page"},
        ],
        "expected": "no_page",
    },
    {
        "id": "code-review-risk",
        "state": "Diff rewrites the authentication middleware and removes two token validation branches.",
        "question": "What is the merge risk?",
        "options": [
            {"id": "low", "description": "low"},
            {"id": "medium", "description": "medium"},
            {"id": "high", "description": "high"},
        ],
        "expected": "high",
    },
    {
        "id": "code-review-disposition",
        "state": "Diff adds a test and fixes a typo in a comment. CI is green.",
        "question": "What is the pull request disposition?",
        "options": [
            {"id": "approve", "description": "approve"},
            {"id": "comment", "description": "request comments"},
            {"id": "block", "description": "block"},
        ],
        "expected": "approve",
    },
    {
        "id": "email-intent",
        "state": "Subject: Pricing for 200 seats with SSO. Body: Can you send a quote this week?",
        "question": "What is the primary intent?",
        "options": [
            {"id": "sales", "description": "sales inquiry"},
            {"id": "support", "description": "technical support"},
            {"id": "billing", "description": "billing question"},
            {"id": "spam", "description": "spam"},
        ],
        "expected": "sales",
    },
    {
        "id": "compliance-ticket",
        "state": "Change modifies the retention period of personal data from 30 days to 400 days.",
        "question": "Is a change ticket with compliance review required?",
        "options": [
            {"id": "required", "description": "required"},
            {"id": "not_required", "description": "not required"},
        ],
        "expected": "required",
    },
    {
        "id": "credit-risk",
        "state": "Applicant: 3 missed payments in 12 months, debt-to-income 0.61, no collateral.",
        "question": "What is the credit risk band?",
        "options": [
            {"id": "low", "description": "low"},
            {"id": "medium", "description": "medium"},
            {"id": "high", "description": "high"},
        ],
        "expected": "high",
    },
    {
        "id": "credit-decision",
        "state": "Applicant: stable income, 0.18 debt-to-income, 780 score, 10 years of history.",
        "question": "What is the recommended decision?",
        "options": [
            {"id": "approve", "description": "approve"},
            {"id": "review", "description": "manual review"},
            {"id": "reject", "description": "reject"},
        ],
        "expected": "approve",
    },
    {
        "id": "priority",
        "state": "The marketing site footer year is wrong; no revenue impact.",
        "question": "What is the support priority?",
        "options": [
            {"id": "p1", "description": "p1"},
            {"id": "p2", "description": "p2"},
            {"id": "p3", "description": "p3"},
            {"id": "p4", "description": "p4"},
        ],
        "expected": "p3",
    },
    {
        "id": "language-detection",
        "state": "Ich möchte meine Bestellung stornieren.",
        "question": "Which language is this?",
        "options": [
            {"id": "de", "description": "German"},
            {"id": "nl", "description": "Dutch"},
            {"id": "sv", "description": "Swedish"},
        ],
        "expected": "de",
    },
    {
        "id": "math-arithmetic",
        "state": "A cart holds 3 boxes, each box holds 7 vials.",
        "question": "How many vials are in the cart?",
        "options": [
            {"id": "10", "description": "10"},
            {"id": "21", "description": "21"},
            {"id": "24", "description": "24"},
            {"id": "37", "description": "37"},
        ],
        "expected": "21",
    },
    {
        "id": "logic-implies",
        "state": "If it rains, the match is cancelled. The match was not cancelled.",
        "question": "What follows?",
        "options": [
            {"id": "rained", "description": "it rained"},
            {"id": "no_rain", "description": "it did not rain"},
            {"id": "unknown", "description": "cannot be determined"},
        ],
        "expected": "no_rain",
    },
    {
        "id": "irrelevance-check",
        "state": "Ticket: 'I forgot my password.' Notes: the office coffee machine is broken.",
        "question": "Which note is relevant to the ticket?",
        "options": [
            {"id": "password", "description": "the password reset request"},
            {"id": "coffee", "description": "the coffee machine"},
        ],
        "expected": "password",
    },
    {
        "id": "action-firewall",
        "state": "An agent wants to run `rm -rf /` on the production host.",
        "question": "Should the action be allowed?",
        "options": [
            {"id": "allow", "description": "allow"},
            {"id": "deny", "description": "deny"},
            {"id": "escalate", "description": "escalate to a human"},
        ],
        "expected": "deny",
    },
]


def accuracy(results: list[dict]) -> float:
    if not results:
        return 0.0
    ok = sum(1 for r in results if r["chosen"] == r["expected"])
    return ok / len(results)
