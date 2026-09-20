# Spexus AI: document Q&A and required message gates

This is a patched edition of [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent),
based on upstream commit `e113c1b3aff3d9248ee1209f4f31cc74f49761c5`.
Upstream authorship and license are retained.

## Companion plugin and setup

The vendor-specific classifier is maintained separately:
**[spexus-ai/hermes-jev-topic-policy](https://github.com/spexus-ai/hermes-jev-topic-policy)**.
Its repository contains the installation instructions, complete example configuration,
topic policy, document corpus, Dockerfile and Compose deployment.

## Patch scope

- Generic `PluginContext.register_message_gate()` API.
- Required async `before_model_request` and `before_response_delivery` checks.
- Fail-closed error handling and immutable request snapshots.
- Opt-in `document_qa` Telegram mode with one tool-free Chat Completions call.
- Stateless questions, bounded corpus/rates/concurrency, no autonomous tools or memory.
- Plain-text delivery only after output approval, without streaming, TTS or media extraction.

See [the gate API documentation](website/docs/developer-guide/plugins/message-gates.md).
The new gates are connected to document mode, **not all Hermes agent entry points**.
Topic classification is supplied by an external plugin and operator policy.

## Verification

The initial implementation passed 76 focused and gateway regression tests, Ruff,
Docker image build and a non-root/read-only container configuration/plugin smoke test.
Network boundaries in the tests use synthetic HTTP responses. The example topic
thresholds are experimental and require calibration against the operator's corpus.

From this checkout, with the plugin repository alongside it:

```sh
uv sync --extra dev --extra messaging
scripts/run_tests.sh tests/agent/test_message_gates.py \
  tests/hermes_cli/test_required_message_gate_plugins.py \
  tests/gateway/test_document_qa.py \
  tests/gateway/test_document_qa_telegram.py \
  ../hermes-jev-topic-policy/tests/test_plugin.py
```
