# Required message gates

Required gates are a separate contract from observer hooks. They are currently
consumed by the opt-in `document_qa` Telegram gateway mode, **not** by the general
AIAgent loop, CLI, API, cron, proxy, or other providers. Enabling `message_gates.required`
without document mode makes gateway startup fail rather than silently omitting checks.

An external plugin registers an async callback for each stage:

```python
from agent.message_gates import GateDecision

async def evaluate(context):
    data = context.payload  # isolated JSON copy, safe to inspect
    if violates_operator_policy(data):
        return GateDecision("deny", "policy_rejected", "off_topic")
    return GateDecision("allow", "policy_allowed")

def register(ctx):
    ctx.register_message_gate("before_model_request", evaluate)
    ctx.register_message_gate("before_response_delivery", evaluate)
```

`context` is an immutable `GateContext` with stage, profile/request/principal/
conversation/destination identities, policy version and serialized payload. Every
read of `payload` returns a fresh copy; plugins cannot mutate the approved request.
Credentials and mutable agent/gateway references are not included. The plugin is
trusted Python code; this API is not a sandbox for malicious installed plugins.

Select callbacks by plugin id:

```yaml
message_gates:
  required:
    before_model_request: [my-policy-plugin]
    before_response_delivery: [my-policy-plugin]
  on_error: deny
  stage_deadline_seconds: 10
  templates:
    off_topic: "Please ask a question about the documented product."
```

Both stages require nonempty lists of registered async callbacks. Configuration is
validated before serving. Ordered chains stop at the first deny. Errors, timeouts,
unloaded callbacks, invalid actions and `None` fail closed. Cancellation propagates.
Only `allow` and `deny` are supported. `template_id` selects a host-owned, static
template; unknown IDs select `unavailable`. No model text is interpolated into refusals.
Static host replies do not recursively invoke classifiers.

The first consumer is `agent/gated_chat.py`: one HTTPS OpenAI-compatible Chat
Completions request, no tools, history, memory, middleware, auxiliary calls, fallback
or output streaming. The request gate sees the exact request object before HTTP
serialization. The output gate sees the final plain text before Telegram send. The
Telegram adapter's text-only path does not extract attachments or enable link previews.

`gateway/document_qa.py` loads a small UTF-8 JSON corpus, validates it, bounds
concurrency/rates and routes queued follow-ups through the same gates. Startup does
not run agent warmup/recovery/cron services. The normal gateway and observer contracts
remain available when document mode is disabled.

Document mode uses these settings:

```yaml
document_qa:
  enabled: true
  corpus_file: documents.json  # relative to active Hermes home
  model_endpoint: https://api.openai.com/v1/chat/completions
  model: YOUR_CHAT_COMPLETIONS_MODEL
  api_key_env: OPENAI_API_KEY
  policy_version: YOUR_APPROVED_POLICY_VERSION
  model_timeout_seconds: 45
  max_concurrent: 4
  requests_per_user_minute: 6
  requests_per_minute: 60
gateway:
  multiplex_profiles: false
group_sessions_per_user: true
thread_sessions_per_user: true
```

Only Telegram may be enabled and profile routing is unsupported. The corpus is an
array of objects with exactly `id`, `title`, `text` (nonempty strings); unique IDs
use `[a-zA-Z0-9_-]{1,64}`. Maximum: 100 documents and 48,000 file bytes. All documents
are included on every request. Responses must cite existing IDs as `[source:ID]`;
this checks citation identity, not factual entailment. Questions are ≤8,192 UTF-8
bytes. Commands, media and internal events cannot start a model turn.

Configuration and corpus are pinned until restart. Use an isolated profile/container
with operator-approved documents suitable for all users and the selected providers.
Adding another consumer/transport requires explicit request and delivery interception
tests; the existence of this plugin API is not a claim of universal coverage.
