# FatSecret Platform API integration

## Scope

ActualPlate currently uses FatSecret's server-side OAuth 2.0 client-credentials flow for application-level food search and food detail lookup. This is not user-profile access and does not imply delegated access to a FatSecret member's diary.

Official references:
- [Authentication](https://platform.fatsecret.com/docs/guides/authentication)
- [Platform API terms](https://platform.fatsecret.com/terms)
- [Attribution policy](https://platform.fatsecret.com/attribution)
- [API editions](https://platform.fatsecret.com/api-editions)

Review these pages again before every commercial release; provider terms, product availability, rate limits, attribution, and data retention rules can change.

## Configuration

Set these variables only in the backend runtime or secret manager:

- `FATSECRET_CLIENT_ID`
- `FATSECRET_CLIENT_SECRET`
- `FATSECRET_TIMEOUT_SECONDS` (optional; default 10 seconds; accepted range 1–60)

Never place credentials in frontend JavaScript, a checked-in `.env`, browser storage, client logs, or public API responses. Rotate credentials through the provider console if exposure is suspected.

The client should not crash unrelated test collection or application imports solely because provider credentials are absent. A request that needs FatSecret must fail with a safe configuration error when credentials are not configured.

## Nutrition data correctness

- Search expressions are bounded and trimmed before sending.
- Provider errors are translated into stable public errors; raw upstream response bodies are not returned to clients.
- OAuth tokens are cached in-process and refreshed before expiry. A 401 response triggers one token refresh/retry; repeated failures are surfaced safely.
- Food records must have a valid gram-based serving before they are used to scale nutrition by estimated mass.
- Invalid, negative, or non-finite required nutrient values are rejected.
- An estimated mass is not supplied by FatSecret. ActualPlate must not represent a serving lookup as a measured portion.
- Ambiguous food matches remain a product limitation: the current integration selects the first provider search result. Before a commercial API contract, evaluate confidence/matching and add explicit selection or a documented deterministic match policy.

## Attribution and display

When FatSecret content is displayed, follow the current official attribution policy. The required HTML attribution snippet must be copied exactly and must not be modified:

```html
<a href="https://platform.fatsecret.com">Powered by fatsecret Platform API</a>
```

The public website includes the attribution link and links to FatSecret's official terms and attribution policy. Verify attribution on every UI surface that renders provider content, including authenticated dashboards and exported reports, before those surfaces ship.

Do not imply ActualPlate is sponsored or endorsed by FatSecret. Do not remove provider notices or required attribution links.

## Storage and caching

Do not add Redis or persistent nutrition-response caching until each relevant content type's current FatSecret storage classification and terms are confirmed. The current terms require content not explicitly identified as indefinitely storable to be removed or refreshed within the provider's specified window (the published terms currently describe 24 hours for content not identified as indefinitely storable). Implement any future cache with explicit TTL, refresh/deletion behavior, and a test proving expiry; never use cache to bypass provider rate limits.

## Failure behavior and observability

- Treat provider 401/403, 429, 5xx, timeout, malformed JSON, missing food, and malformed serving data as distinct internal categories.
- Public responses contain safe messages and a provider identifier only; logs may record status category and correlation ID but must not log secrets or raw provider payloads.
- Add retries only for safe, bounded cases; never retry a 429 aggressively.
- Use mocked contract tests in ordinary CI. Real credentialed integration tests must be opt-in and run only in a protected environment with authorized credentials.
- Monitor provider error rate and request volume before introducing automated caching or fallback providers.

## Release checklist

- [ ] Verify credentials in deployment secret storage.
- [ ] Confirm provider account/IP restrictions and allowed markets.
- [ ] Confirm the selected FatSecret edition permits the intended commercial use and data display.
- [ ] Verify required attribution on the website and every provider-content surface.
- [ ] Verify data retention, refresh, and deletion rules for each stored field.
- [ ] Run mocked contract tests and a protected live smoke test.
- [ ] Verify provider failures never leak response bodies or credentials.
- [ ] Record the reviewed date and the exact terms/policy revision in release evidence.
