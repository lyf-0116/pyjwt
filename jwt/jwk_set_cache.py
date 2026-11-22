import math
import random
import time
from collections import OrderedDict
from typing import Any, Callable, Optional, Union

from .api_jwk import PyJWK, PyJWKSet, PyJWTSetWithTimestamp
from .exceptions import PyJWKSetError


class JWKSetCache:
    def __init__(
        self,
        lifespan: float,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self.jwk_set_with_timestamp: Optional[PyJWTSetWithTimestamp] = None
        self.lifespan = lifespan
        self._clock = clock if clock is not None else time.monotonic

    def put(self, jwk_set: Optional[Union[PyJWKSet, dict[str, Any]]]) -> None:
        """Cache *jwk_set*, or clear the cache when it is ``None``.

        Accepts either an already-parsed ``PyJWKSet`` or the raw JWKS
        payload it is parsed from, and always stores it parsed so that
        reads never have to parse the keys again.

        :raises PyJWKSetError: If *jwk_set* is neither a ``PyJWKSet`` nor a
            JWKS payload containing usable keys.
        """
        if jwk_set is None:
            # clear cache
            self.jwk_set_with_timestamp = None
            return

        if not isinstance(jwk_set, PyJWKSet):
            if not isinstance(jwk_set, dict):
                raise PyJWKSetError("Invalid JWK Set value")
            jwk_set = PyJWKSet.from_dict(jwk_set)

        self.jwk_set_with_timestamp = PyJWTSetWithTimestamp(jwk_set, self._clock)

    def get(self) -> Optional[PyJWKSet]:
        if self.jwk_set_with_timestamp is None or self.is_expired():
            return None

        return self.jwk_set_with_timestamp.get_jwk_set()

    def is_expired(self) -> bool:
        return (
            self.jwk_set_with_timestamp is not None
            and self.lifespan > -1
            and self._clock()
            > self.jwk_set_with_timestamp.get_timestamp() + self.lifespan
        )


class SigningKeyCache:
    """TTL-aware LRU cache for individual signing keys.

    The JWK Set cache (:class:`JWKSetCache`) expires entries after a
    lifespan. A plain :func:`functools.lru_cache` has no time-based
    expiration, so a key it keeps can outlive the key set it came from:
    lookups either trust a stale key (while the entry is present) or hit
    the network again once LRU pressure has evicted it. This cache aligns
    tier-2 semantics with tier-1 by expiring every entry after the same
    lifespan while keeping LRU eviction for the bounded size.
    """

    def __init__(
        self,
        maxsize: int,
        lifespan: float,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        if maxsize <= 0:
            raise PyJWKSetError("Signing key cache maxsize must be greater than 0")
        if lifespan <= 0:
            raise PyJWKSetError("Signing key cache lifespan must be greater than 0")
        self.maxsize = maxsize
        self.lifespan = lifespan
        self._clock = clock if clock is not None else time.monotonic
        self._entries: OrderedDict[str, tuple[PyJWK, float]] = OrderedDict()

    def get(self, kid: str) -> Optional[PyJWK]:
        entry = self._entries.get(kid)
        if entry is None:
            return None
        signing_key, cached_at = entry
        if self._clock() > cached_at + self.lifespan:
            del self._entries[kid]
            return None
        self._entries.move_to_end(kid)
        return signing_key

    def put(self, kid: str, signing_key: PyJWK) -> None:
        self._entries[kid] = (signing_key, self._clock())
        self._entries.move_to_end(kid)
        while len(self._entries) > self.maxsize:
            self._entries.popitem(last=False)

    def clear(self) -> None:
        self._entries.clear()


class _KidRefreshState:
    """Per-source refresh bookkeeping used by :class:`RefreshRateLimiter`."""

    __slots__ = ("last_attempt", "failures")

    def __init__(self, last_attempt: float, failures: int = 0) -> None:
        self.last_attempt = last_attempt
        self.failures = failures


def _default_jitter(value: float) -> float:
    return random.uniform(0, value)


class RefreshRateLimiter:
    """Source-aware rate limiter for forced JWKS refreshes.

    A single global time window is an amplification vector: every distinct
    (forged) ``kid`` gets one outbound request per window, and repeated
    lookups of the same ID cannot be distinguished from new ones. This
    limiter keeps two layers instead:

    - A **per-kid cooldown**: repeated lookups of the same unknown key ID
      do not trigger another fetch until the cooldown elapses, even if the
      ID was evicted from other caches.
    - A **global floor**: regardless of how many distinct key IDs are
      presented, forced refreshes are at most one per cooldown window
      across the whole client, so a flood of forged IDs cannot fan out
      into a flood of outbound requests.

    Failed refreshes use exponential backoff (doubled per consecutive
    failure, capped) plus jitter, so an unhealthy endpoint is not hit in
    lock-step by many clients and a key ID that keeps failing backs off
    further on each failure. A successful refresh resets the backoff.
    """

    #: Upper bound on the backoff applied after repeated failures.
    MAX_BACKOFF = 3600.0

    def __init__(
        self,
        cooldown_duration: float,
        max_tracked_kids: int,
        *,
        jitter: Optional[Callable[[float], float]] = None,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        if cooldown_duration < 0:
            raise PyJWKSetError(
                "Cooldown duration must be greater than or equal to 0, "
                f'the input is "{cooldown_duration}"'
            )
        if not math.isfinite(cooldown_duration):
            raise PyJWKSetError(
                f'Cooldown duration must be finite, the input is "{cooldown_duration}"'
            )
        self.cooldown_duration = cooldown_duration
        self._jitter: Callable[[float], float] = (
            jitter if jitter is not None else _default_jitter
        )
        self._clock = clock if clock is not None else time.monotonic
        self.max_tracked_kids = max_tracked_kids
        self._states: OrderedDict[str, _KidRefreshState] = OrderedDict()

    def _backoff_for(self, failures: int) -> float:
        if failures <= 0:
            return 0.0
        backoff: float = min(
            self.cooldown_duration * (2 ** (failures - 1)), self.MAX_BACKOFF
        )
        jitter: float = self._jitter(backoff)
        return backoff + jitter

    def _retain(self, kid: str, state: _KidRefreshState) -> None:
        self._states[kid] = state
        self._states.move_to_end(kid)
        while len(self._states) > self.max_tracked_kids:
            self._states.popitem(last=False)

    def is_allowed(
        self,
        kid: str,
        global_floor: float,
        now: Optional[float] = None,
    ) -> bool:
        """Return whether a forced refresh for *kid* may hit the network.

        *global_floor* is the timestamp of the most recent successful
        fetch of any kind; the client owns it so its existing
        ``_last_successful_fetch`` attribute remains the single source of
        truth for the cross-kid rate floor.
        """
        if self.cooldown_duration == 0:
            return True
        current = self._clock() if now is None else now
        if current - global_floor < self.cooldown_duration:
            return False

        state = self._states.get(kid)
        if state is None:
            return True
        return current - state.last_attempt >= self._backoff_for(state.failures)

    def record_attempt(self, kid: str, now: Optional[float] = None) -> None:
        """Record that a forced refresh for *kid* is starting."""
        current = self._clock() if now is None else now
        state = self._states.get(kid)
        if state is None:
            state = _KidRefreshState(current)
        else:
            state.last_attempt = current
        self._retain(kid, state)

    def record_success(self, kid: str) -> None:
        """Reset the per-kid failure backoff after a successful refresh."""
        state = self._states.get(kid)
        if state is not None:
            state.failures = 0

    def record_failure(self, kid: str) -> None:
        """Grow the per-kid exponential backoff after a failed refresh."""
        state = self._states.get(kid)
        if state is not None:
            state.failures += 1
