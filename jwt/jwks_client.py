from __future__ import annotations

import http.client
import json
import math
import threading
import time
import urllib.request
from ssl import SSLContext
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse

from .api_jwk import PyJWK, PyJWKSet
from .api_jwt import decode_complete as decode_token
from .exceptions import PyJWKClientConnectionError, PyJWKClientError
from .jwk_set_cache import JWKSetCache, RefreshRateLimiter, SigningKeyCache


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        return None


class PyJWKClient:
    def __init__(
        self,
        uri: str,
        cache_keys: bool = False,
        max_cached_keys: int = 16,
        cache_jwk_set: bool = True,
        lifespan: float = 300,
        headers: dict[str, Any] | None = None,
        timeout: float = 30,
        ssl_context: SSLContext | None = None,
        cooldown_duration: float = 30,
    ):
        """A client for retrieving signing keys from a JWKS endpoint.

        ``PyJWKClient`` uses a two-tier caching system to avoid unnecessary
        network requests:

        **Tier 1 — JWK Set cache** (enabled by default):
        Caches the entire JSON Web Key Set response from the endpoint.
        Controlled by:

        - ``cache_jwk_set``: Set to ``True`` (the default) to enable this
          cache. When enabled, the JWK Set is fetched from the network only
          when the cache is empty or expired.
        - ``lifespan``: Time in seconds before the cached JWK Set expires.
          Defaults to ``300`` (5 minutes). Must be greater than 0.

        Unknown key IDs can trigger a forced refresh, gated by a
        source-aware rate limiter configured through ``cooldown_duration``.
        The limiter tracks each key ID separately, so repeated lookups of
        the same unknown ID do not cause repeated fetches, while a global
        floor caps forced refreshes at one per cooldown window across the
        whole client: a flood of distinct forged IDs cannot fan out into a
        flood of outbound requests. Failed refreshes back off
        exponentially (with jitter) per key ID; a successful refresh resets
        that backoff. Set ``cooldown_duration`` to ``0`` to disable
        limiting. The limiter is bypassed when ``cache_jwk_set`` is
        ``False``.

        **Tier 2 — Signing key cache** (disabled by default):
        Caches individual signing keys (looked up by ``kid``) in a
        bounded LRU cache whose entries expire after the same ``lifespan``
        as the Tier 1 JWK Set cache. Keys are evicted early only when the
        cache reaches its maximum size. Controlled by:

        - ``cache_keys``: Set to ``True`` to enable this cache.
          Defaults to ``False``.
        - ``max_cached_keys``: Maximum number of signing keys to keep in
          the LRU cache. Defaults to ``16``. The entries expire after the
          same ``lifespan`` as the JWK Set cache, so a key can never
          outlive the key set it came from, and eviction no longer turns a
          previously-cached ID into a network request on its next lookup.

        :param uri: The URL of the JWKS endpoint.
        :type uri: str
        :param cache_keys: Enable the per-key LRU cache (Tier 2).
        :type cache_keys: bool
        :param max_cached_keys: Max entries in the signing key LRU cache.
        :type max_cached_keys: int
        :param cache_jwk_set: Enable the JWK Set response cache (Tier 1).
        :type cache_jwk_set: bool
        :param lifespan: TTL in seconds for the JWK Set cache.
        :type lifespan: float
        :param headers: Optional HTTP headers to include in requests.
        :type headers: dict or None
        :param timeout: HTTP request timeout in seconds.
        :type timeout: float
        :param ssl_context: Optional SSL context for the request.
        :type ssl_context: ssl.SSLContext or None
        :param cooldown_duration: Minimum time in seconds between forced
            refreshes, applied per key ID and as a global floor. Failed
            refreshes extend this with exponential backoff and jitter.
            Defaults to ``30``; set to ``0`` to disable limiting.
        :type cooldown_duration: float
        """
        if headers is None:
            headers = {}
        # urllib's default OpenerDirector also handles file://, ftp://, and
        # data: URIs. Reject anything that isn't http(s) eagerly so a caller
        # passing an attacker-influenced URL (e.g. taken from a `jku` token
        # header) can't read local files or reach other unintended schemes.
        scheme = urlparse(uri).scheme.lower()
        if scheme not in ("http", "https"):
            raise PyJWKClientError(
                f"Invalid JWKS URI scheme {scheme!r}: only 'http' and 'https' "
                f"are supported."
            )
        self.uri = uri
        self.jwk_set_cache: JWKSetCache | None = None
        self.headers = headers
        self.timeout = timeout
        self.ssl_context = ssl_context
        if cooldown_duration < 0:
            raise PyJWKClientError(
                "Cooldown duration must be greater than or equal to 0, "
                f'the input is "{cooldown_duration}"'
            )
        if not math.isfinite(cooldown_duration):
            raise PyJWKClientError(
                f'Cooldown duration must be finite, the input is "{cooldown_duration}"'
            )
        self.cooldown_duration = cooldown_duration
        self._last_successful_fetch: float | None = None
        self._client_lock = threading.RLock()
        self._refresh_limiter: RefreshRateLimiter | None = None
        self._signing_key_cache: SigningKeyCache | None = None

        # Both tiers share the lifespan so their hit/expiry semantics stay
        # aligned: the per-key cache must not serve keys the key-set cache
        # would already consider stale.
        if cache_jwk_set or cache_keys:
            if lifespan <= 0:
                raise PyJWKClientError(
                    f'Lifespan must be greater than 0, the input is "{lifespan}"'
                )

        if cache_jwk_set:
            # Init jwt set cache with default or given lifespan.
            # Default lifespan is 300 seconds (5 minutes).
            self.jwk_set_cache = JWKSetCache(lifespan, clock=self._now)
            self._refresh_limiter = RefreshRateLimiter(
                cooldown_duration, max_cached_keys, clock=self._now
            )
        else:
            self.jwk_set_cache = None

        if cache_keys:
            # Cache signing keys with the same TTL as the JWK Set cache.
            self._signing_key_cache = SigningKeyCache(
                max_cached_keys, lifespan, clock=self._now
            )

    def _now(self) -> float:
        # Indirection so tests can advance a single fake clock via
        # `jwt.jwks_client.time.monotonic` and drive every time-based
        # decision (limiter, caches) from it.
        return time.monotonic()

    def fetch_data(self) -> Any:
        """Fetch the JWK Set from the JWKS endpoint.

        Makes an HTTP request to the configured ``uri`` and returns the
        parsed JSON response. If the JWK Set cache is enabled, the
        response is stored in the cache.

        :returns: The parsed JWK Set as a dictionary.
        :raises PyJWKClientConnectionError: If the HTTP request fails.
        :raises PyJWKClientError: If the endpoint does not return a JSON
            object.
        :raises PyJWKSetError: If the JWK Set cache is enabled and the
            response contains no usable keys.
        """
        try:
            r = urllib.request.Request(url=self.uri, headers=self.headers)
            handlers: list[Any] = [_NoRedirectHandler()]
            if self.ssl_context is not None:
                handlers.append(urllib.request.HTTPSHandler(context=self.ssl_context))
            opener = urllib.request.build_opener(*handlers)
            with opener.open(r, timeout=self.timeout) as response:
                jwk_set = json.load(response)
        except (URLError, TimeoutError, http.client.HTTPException) as e:
            if isinstance(e, HTTPError):
                e.close()
            raise PyJWKClientConnectionError(
                f'Fail to fetch data from the url, err: "{e}"'
            ) from e

        # Validate the payload before it reaches the cache, so an endpoint
        # returning something that isn't a JSON object is reported as such
        # rather than as a malformed JWK Set.
        jwk_set = self._as_jwk_set_payload(jwk_set)

        # Only update the cache on a successful fetch. Writing in a
        # `finally` block with `jwk_set=None` on error clears any
        # previously-cached JWKS, turning a transient outage into a cache
        # wipe that breaks legitimate auth.
        if self.jwk_set_cache is not None:
            self.jwk_set_cache.put(jwk_set)
        self._last_successful_fetch = time.monotonic()
        return jwk_set

    def get_jwk_set(self, refresh: bool = False) -> PyJWKSet:
        """Return the JWK Set, using the cache when available.

        :param refresh: Force a fresh fetch from the endpoint, bypassing
            the cache.
        :type refresh: bool
        :returns: The JWK Set.
        :rtype: PyJWKSet
        :raises PyJWKClientError: If the endpoint does not return a JSON
            object.
        """
        data = None
        fetched = False
        if self.jwk_set_cache is not None and not refresh:
            data = self.jwk_set_cache.get()

        if data is None:
            data = self.fetch_data()
            fetched = True

        # A cache hit is already parsed, so serve it as-is. Only a fresh
        # fetch reaches the check below, which still matters because
        # `fetch_data()` may be overridden by a subclass.
        if isinstance(data, PyJWKSet):
            jwk_set = data
        else:
            jwk_set = PyJWKSet.from_dict(self._as_jwk_set_payload(data))

        # `fetch_data()` caches the payload it received from the endpoint,
        # but a subclass may filter or transform it before returning. Cache
        # the value actually being returned, so later cache hits serve the
        # same key set as this call rather than the pre-transform one.
        if fetched and self.jwk_set_cache is not None:
            self.jwk_set_cache.put(jwk_set)

        return jwk_set

    @staticmethod
    def _as_jwk_set_payload(data: Any) -> dict[str, Any]:
        if not isinstance(data, dict):
            raise PyJWKClientError("The JWKS endpoint did not return a JSON object")

        return data

    def get_signing_keys(self, refresh: bool = False) -> list[PyJWK]:
        """Return all signing keys from the JWK Set.

        Filters the JWK Set to keys whose ``use`` is ``"sig"`` (or
        unspecified) and that have a ``kid``.

        :param refresh: Force a fresh fetch from the endpoint, bypassing
            the cache.
        :type refresh: bool
        :returns: A list of signing keys.
        :rtype: list[PyJWK]
        :raises PyJWKClientError: If no signing keys are found.
        """
        jwk_set = self.get_jwk_set(refresh)
        return self._get_signing_keys_from_jwk_set(jwk_set)

    @staticmethod
    def _get_signing_keys_from_jwk_set(jwk_set: PyJWKSet) -> list[PyJWK]:
        signing_keys = [
            jwk_set_key
            for jwk_set_key in jwk_set.keys
            if jwk_set_key.public_key_use in ["sig", None] and jwk_set_key.key_id
        ]

        if not signing_keys:
            raise PyJWKClientError("The JWKS endpoint did not contain any signing keys")

        return signing_keys

    def get_signing_key(self, kid: str) -> PyJWK:
        """Return the signing key matching the given ``kid``.

        If no match is found in the current JWK Set (or the per-key cache),
        the set is refreshed from the endpoint and the lookup is retried
        once when the source-aware refresh limiter permits it. Repeated
        misses for the same ``kid`` are cooled down per key ID, a global
        floor limits distinct forged IDs to one refresh per window, and
        failed refreshes back off exponentially with jitter.

        :param kid: The key ID to look up.
        :type kid: str
        :returns: The matching signing key.
        :rtype: PyJWK
        :raises PyJWKClientError: If no matching key is found after
            refreshing.
        """
        with self._client_lock:
            cached_key = (
                self._signing_key_cache.get(kid)
                if self._signing_key_cache is not None
                else None
            )
            if cached_key is not None:
                return cached_key

            signing_keys = self.get_signing_keys()
            signing_key = self.match_kid(signing_keys, kid)

            if not signing_key:
                limiter = self._refresh_limiter
                global_floor = (
                    self._last_successful_fetch
                    if self._last_successful_fetch is not None
                    else 0.0
                )
                if limiter is None or limiter.is_allowed(kid, global_floor):
                    if limiter is not None:
                        limiter.record_attempt(kid)
                    try:
                        signing_keys = self.get_signing_keys(refresh=True)
                    except PyJWKClientError:
                        # A failed fetch must count against the per-kid
                        # budget with exponential backoff + jitter instead
                        # of being retried immediately on the next call.
                        if limiter is not None:
                            limiter.record_failure(kid)
                        raise
                    if limiter is not None:
                        limiter.record_success(kid)
                    signing_key = self.match_kid(signing_keys, kid)

                if not signing_key:
                    raise PyJWKClientError(
                        f'Unable to find a signing key that matches: "{kid}"'
                    )

            if self._signing_key_cache is not None:
                self._signing_key_cache.put(kid, signing_key)
            return signing_key

    def get_signing_key_from_jwt(self, token: str | bytes) -> PyJWK:
        """Return the signing key for a JWT by reading its ``kid`` header.

        Extracts the ``kid`` from the token's unverified header and
        delegates to :meth:`get_signing_key`.

        :param token: The encoded JWT.
        :type token: str or bytes
        :returns: The matching signing key.
        :rtype: PyJWK
        """
        unverified = decode_token(token, options={"verify_signature": False})
        header = unverified["header"]
        return self.get_signing_key(header.get("kid"))

    @staticmethod
    def match_kid(signing_keys: list[PyJWK], kid: str) -> PyJWK | None:
        """Find a key in *signing_keys* that matches *kid*.

        :param signing_keys: The list of keys to search.
        :type signing_keys: list[PyJWK]
        :param kid: The key ID to match.
        :type kid: str
        :returns: The matching key, or ``None`` if not found.
        :rtype: PyJWK or None
        """
        signing_key = None

        for key in signing_keys:
            if key.key_id == kid:
                signing_key = key
                break

        return signing_key
