"""The token hash, the bearer-token check, and the rate limiter."""

from __future__ import annotations

import hashlib
import hmac
import re
import threading
import time
from typing import Any

import pytest

from seeingmon.clock import VirtualClock
from seeingmon.services.web.auth import (
    AuthConfigError,
    RateLimiter,
    ScryptParams,
    TokenVerifier,
    bearer_token,
    hash_token,
    parse_token_hash,
    verify_token,
)

TOKEN = "t0ken-for-the-tests-QRSTUVWXYZ-ghijklmn"
CHEAP = ScryptParams(ln=10, r=8, p=1)  # the tests do not need a hash that costs 32 MiB


@pytest.fixture
def stored() -> str:
    return hash_token(TOKEN, params=CHEAP)


def count_scrypt(monkeypatch: pytest.MonkeyPatch) -> list[bytes]:
    """Record each candidate that reaches scrypt, and let the real function run."""
    seen: list[bytes] = []
    real = hashlib.scrypt

    def spy(password: bytes, **kwargs: Any) -> bytes:
        seen.append(password)
        return real(password, **kwargs)

    monkeypatch.setattr(hashlib, "scrypt", spy)
    return seen


# --- The hash format -------------------------------------------------------------------------


def test_a_hash_follows_the_documented_format(stored: str) -> None:
    pattern = r"\$scrypt\$ln=10,r=8,p=1\$[A-Za-z0-9+/]{22}\$[A-Za-z0-9+/]{43}"
    assert re.fullmatch(pattern, stored), stored


def test_the_default_parameters_cost_32_mib() -> None:
    params = ScryptParams()
    assert (params.ln, params.r, params.p) == (15, 8, 1)
    assert 32 * 1024 * 1024 <= params.memory_bytes < 33 * 1024 * 1024


def test_a_default_hash_verifies() -> None:
    """The default cost needs a `maxmem` above the 32 MiB that OpenSSL allows by default."""
    stored = hash_token(TOKEN)
    assert stored.startswith("$scrypt$ln=15,r=8,p=1$")
    assert verify_token(TOKEN, stored)
    assert not verify_token(TOKEN + "x", stored)


def test_two_hashes_of_one_token_differ_because_the_salt_differs() -> None:
    assert hash_token(TOKEN, params=CHEAP) != hash_token(TOKEN, params=CHEAP)


def test_the_hash_never_contains_the_token(stored: str) -> None:
    assert TOKEN not in stored


def test_the_hash_records_its_parameters_so_an_old_hash_keeps_working() -> None:
    cheaper = hash_token(TOKEN, params=ScryptParams(ln=9, r=8, p=2))
    assert verify_token(TOKEN, cheaper)
    parsed = parse_token_hash(cheaper)
    assert (parsed.params.ln, parsed.params.r, parsed.params.p) == (9, 8, 2)
    assert len(parsed.salt) == 16
    assert len(parsed.digest) == 32


def test_a_hash_with_a_given_salt_is_reproducible() -> None:
    first = hash_token(TOKEN, params=CHEAP, salt=b"sixteen-bytesalt")
    assert first == hash_token(TOKEN, params=CHEAP, salt=b"sixteen-bytesalt")


def test_the_right_token_verifies_and_a_wrong_one_does_not(stored: str) -> None:
    assert verify_token(TOKEN, stored)
    assert not verify_token(TOKEN.upper(), stored)
    assert not verify_token("", stored)
    assert not verify_token("\ud800", stored)  # a lone surrogate cannot be encoded


@pytest.mark.parametrize("token", ["", "x" * 257])
def test_a_token_must_have_1_to_256_characters(token: str) -> None:
    with pytest.raises(AuthConfigError, match="1 to 256"):
        hash_token(token, params=CHEAP)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "plain",
        "$argon2$ln=10,r=8,p=1$c2FsdHNhbHRzYWx0$ZGlnZXN0ZGlnZXN0ZGlnZXN0",
        "$scrypt$ln=10,r=8$c2FsdHNhbHRzYWx0$ZGlnZXN0ZGlnZXN0ZGlnZXN0",
        "$scrypt$ln=10,r=8,p=1$c2FsdHNhbHRzYWx0",
        "$scrypt$ln=3,r=8,p=1$c2FsdHNhbHRzYWx0$ZGlnZXN0ZGlnZXN0ZGlnZXN0",
        "$scrypt$ln=30,r=8,p=1$c2FsdHNhbHRzYWx0$ZGlnZXN0ZGlnZXN0ZGlnZXN0",
        "$scrypt$ln=20,r=32,p=16$c2FsdHNhbHRzYWx0$ZGlnZXN0ZGlnZXN0ZGlnZXN0",
        "$scrypt$ln=10,r=8,p=1$!!!$ZGlnZXN0ZGlnZXN0ZGlnZXN0",
        "$scrypt$ln=10,r=8,p=1$c2FsdA$ZGlnZXN0ZGlnZXN0ZGlnZXN0",
        "$scrypt$ln=10,r=8,p=1$c2FsdHNhbHRzYWx0$ZGln",
        "<hash of the API token>",
    ],
)
def test_a_malformed_hash_is_refused_without_echoing_it(text: str) -> None:
    with pytest.raises(AuthConfigError) as raised:
        parse_token_hash(text)
    if text:
        assert text not in str(raised.value)


def test_the_scrypt_parameters_have_bounds() -> None:
    for ln, r, p in [(3, 8, 1), (21, 8, 1), (10, 0, 1), (10, 33, 1), (10, 8, 0), (10, 8, 17)]:
        with pytest.raises(AuthConfigError):
            ScryptParams(ln, r, p)


# --- The Authorization header ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        (f"Bearer {TOKEN}", TOKEN),
        (f"bearer {TOKEN}", TOKEN),
        (f"BEARER   {TOKEN}  ", TOKEN),
        ("Bearer abc.def-ghi_jkl~mno+pqr/stu=", "abc.def-ghi_jkl~mno+pqr/stu="),
        (None, None),
        ("", None),
        ("Bearer", None),
        ("Bearer ", None),
        (f"Basic {TOKEN}", None),
        (TOKEN, None),
        ("Bearer a b", None),
        ("Bearer a\x00b", None),
        ("Bearer é", None),
        ("Bearer " + "x" * 257, None),
        ("Bearer " + "x" * 5000, None),
        ("Bearer token,other", None),
    ],
)
def test_bearer_token_reads_only_a_well_formed_header(
    header: str | None, expected: str | None
) -> None:
    assert bearer_token(header) == expected


# --- The verifier ----------------------------------------------------------------------------


def make_verifier(stored: str | None, clock: VirtualClock | None = None) -> TokenVerifier:
    return TokenVerifier(stored, clock or VirtualClock(), cache_ttl_s=60.0)


def test_a_verifier_without_a_hash_accepts_nothing() -> None:
    verifier = make_verifier(None)
    assert not verifier.enabled
    assert not verifier.verify(TOKEN)
    assert not verifier.verify("")


def test_a_verifier_accepts_the_token_and_rejects_every_other_candidate(stored: str) -> None:
    verifier = make_verifier(stored)
    assert verifier.enabled
    assert verifier.verify(TOKEN)
    for candidate in ["", "x", TOKEN[:-1], TOKEN + "x", "x" * 300, "\ud800", "Bearer " + TOKEN]:
        assert not verifier.verify(candidate)


def test_a_verifier_refuses_a_malformed_hash_at_construction() -> None:
    with pytest.raises(AuthConfigError):
        make_verifier("<hash of the API token>")


def test_every_wrong_candidate_costs_one_scrypt_run_whatever_its_length(
    stored: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wrong token must not return early, or the time would tell how close it was."""
    seen = count_scrypt(monkeypatch)
    verifier = make_verifier(stored)
    candidates = ["a", "ab" * 10, "x" * 256, TOKEN[:-1], TOKEN + "!"]
    for candidate in candidates:
        assert not verifier.verify(candidate)
    assert seen == [candidate.encode() for candidate in candidates]


def test_the_digests_are_compared_in_constant_time(
    stored: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each check calls `hmac.compare_digest` on two digests of the same length."""
    calls: list[tuple[int, int]] = []
    real = hmac.compare_digest

    def spy(a: bytes, b: bytes) -> bool:
        calls.append((len(a), len(b)))
        return real(a, b)

    monkeypatch.setattr(hmac, "compare_digest", spy)
    verifier = make_verifier(stored)
    assert not verifier.verify("short")
    assert not verifier.verify("a-much-longer-wrong-candidate-than-the-first")
    assert verifier.verify(TOKEN)
    assert calls
    assert set(calls) == {(32, 32)}


def test_a_passing_token_is_cached_and_a_failing_one_is_not(
    stored: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = count_scrypt(monkeypatch)
    verifier = make_verifier(stored)
    assert verifier.verify(TOKEN)
    assert verifier.verify(TOKEN)
    assert verifier.verify(TOKEN)
    assert len(seen) == 1
    assert not verifier.verify("wrong-one")
    assert not verifier.verify("wrong-one")
    assert len(seen) == 3  # the first pass, and one run for each failure


def test_the_cache_expires(stored: str, monkeypatch: pytest.MonkeyPatch) -> None:
    clock = VirtualClock()
    seen = count_scrypt(monkeypatch)
    verifier = make_verifier(stored, clock)
    assert verifier.verify(TOKEN)
    clock.advance(59)
    assert verifier.verify(TOKEN)
    assert len(seen) == 1
    clock.advance(2)
    assert verifier.verify(TOKEN)
    assert len(seen) == 2


def test_the_verifier_remembers_a_digest_and_never_the_token(stored: str) -> None:
    verifier = make_verifier(stored)
    assert verifier.verify(TOKEN)
    remembered = verifier._passed_key  # the test looks inside on purpose
    assert len(remembered) == 32
    assert remembered != bytes(32)
    assert TOKEN.encode() not in remembered


def test_only_one_scrypt_runs_at_a_time(stored: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """A flood of wrong tokens must not use more than one hash worth of memory."""
    active = 0
    peak = 0
    lock = threading.Lock()
    real = hashlib.scrypt

    def slow(password: bytes, **kwargs: Any) -> bytes:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            time.sleep(0.002)
            return real(password, **kwargs)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(hashlib, "scrypt", slow)
    verifier = make_verifier(stored)
    threads = [
        threading.Thread(target=verifier.verify, args=(f"wrong-{index}",)) for index in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert peak == 1


# --- The rate limiter ------------------------------------------------------------------------


def test_a_limiter_allows_the_limit_and_refuses_the_next_event() -> None:
    clock = VirtualClock()
    limiter = RateLimiter(clock, limit=3, window_s=10.0)
    assert [limiter.acquire("a") for _ in range(3)] == [None, None, None]
    assert limiter.acquire("a") == 10
    clock.advance(4)
    assert limiter.acquire("a") == 6  # the oldest event leaves the window in 6 s
    clock.advance(6)
    assert limiter.acquire("a") is None  # the first three events left at exactly 10 s


def test_a_refused_event_is_not_counted() -> None:
    clock = VirtualClock()
    limiter = RateLimiter(clock, limit=1, window_s=10.0)
    assert limiter.acquire("a") is None
    for _ in range(5):
        assert limiter.acquire("a") is not None
    clock.advance(10)
    assert limiter.acquire("a") is None  # the refusals did not extend the block


def test_the_clients_have_separate_limits() -> None:
    limiter = RateLimiter(VirtualClock(), limit=1, window_s=10.0)
    assert limiter.acquire("a") is None
    assert limiter.acquire("b") is None
    assert limiter.acquire("a") is not None
    assert limiter.acquire("b") is not None


def test_the_retry_time_rounds_up_to_whole_seconds() -> None:
    clock = VirtualClock()
    limiter = RateLimiter(clock, limit=1, window_s=10.0)
    assert limiter.acquire("a") is None
    clock.advance(9.2)
    assert limiter.acquire("a") == 1  # 0.8 s remain


def test_check_and_record_count_only_what_the_caller_records() -> None:
    clock = VirtualClock()
    limiter = RateLimiter(clock, limit=2, window_s=60.0)
    assert limiter.check("a") is None
    limiter.record("a")
    assert limiter.check("a") is None
    limiter.record("a")
    assert limiter.check("a") == 60
    clock.advance(60)
    assert limiter.check("a") is None


def test_the_limiter_forgets_the_least_recent_client_beyond_its_bound() -> None:
    clock = VirtualClock()
    limiter = RateLimiter(clock, limit=5, window_s=60.0, max_clients=3)
    for name in ["a", "b", "c", "d"]:
        limiter.record(name)
    assert limiter.clients == 3
    assert limiter.check("a") is None  # forgotten
    for _ in range(4):
        limiter.record("d")
    assert limiter.check("d") is not None


def test_the_limiter_drops_idle_clients_first() -> None:
    clock = VirtualClock()
    limiter = RateLimiter(clock, limit=5, window_s=10.0, max_clients=100)
    for index in range(100):
        limiter.record(f"old-{index}")
    clock.advance(11)
    limiter.record("new")
    assert limiter.clients == 1  # the new client pushed the table over its bound, so the idle left
    for index in range(150):
        limiter.record(f"fresh-{index}")
    assert limiter.clients == 100  # and the bound holds when every client is active


@pytest.mark.parametrize(
    "arguments", [{"limit": 0, "window_s": 1.0}, {"limit": 1, "window_s": 0.0}]
)
def test_the_limiter_rejects_nonsense(arguments: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="positive"):
        RateLimiter(VirtualClock(), **arguments)


def test_a_limiter_can_be_shared_by_threads() -> None:
    limiter = RateLimiter(VirtualClock(), limit=50, window_s=60.0)
    outcomes: list[int | None] = []
    lock = threading.Lock()

    def work() -> None:
        for _ in range(20):
            result = limiter.acquire("shared")
            with lock:
                outcomes.append(result)

    threads: list[threading.Thread] = [threading.Thread(target=work) for _ in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sum(1 for outcome in outcomes if outcome is None) == 50
    assert sum(1 for outcome in outcomes if outcome is not None) == 50
