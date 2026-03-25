# Voice Agents

AI voice agent demo built on LangGraph with TrustyAI Guardrails integration.

## Project Structure

```
ai-voice-agent/
  backend/          Python WebSocket server (LangGraph agent graph)
  frontend/         React/Next.js UI
  deploy/chart/     Helm chart for OpenShift deployment
  guardrails/       Kustomize resources for Guardrails
```

## Guardrails

The app supports two independent guardrails systems that can be toggled on/off independently via the UI:

- **FMS** (TrustyAI Guardrails Orchestrator) — env var `GUARDRAILS_URL`
- **NeMo** (NeMo Guardrails server) — env var `NEMO_GUARDRAILS_URL`

The `guardrails_mode` is one of `"none"`, `"fms"`, `"nemo"`, or `"both"`. Four pre-compiled graphs are built at startup (one per mode). The frontend sends `set_guardrails_mode` messages and the backend selects the corresponding graph.

### FMS Guardrails

When `GUARDRAILS_URL` is set, the backend creates `ChatOpenAI` instances pointed at the orchestrator via the nginx proxy. Detectors are passed in `extra_body`. A custom `httpx` event hook logs detection results and warnings from every orchestrator response to both stdout and MLFlow traces (via `threading.local()` → `mlflow.get_current_active_span().set_attribute()`).

#### Orchestrator limitations

- **No streaming** — returns empty response for `stream: true`. All guardrails LLMs use `streaming=False`.
- **No `tool` role messages** — rejects with 422. Agent nodes use regular agents (with tools, regular LLM) and screen output separately.

#### FMS Screening flow

1. **`_screen_user_input`** — pre-screens the user's raw message with all four input detectors (`GUARDRAILS_DETECTORS_INPUT_ONLY`) before supervisor routing. Only the single user message is sent to avoid false positives from system prompts.
2. **Supervisor routing** — regular LLM with structured output (no guardrails).
3. **Supervisor direct response** (route=none) — `guardrails_llm` with full input+output detectors (`GUARDRAILS_DETECTORS`).
4. **Agent nodes** (pizza, order, delivery) — regular agents with tools (regular LLM). Orchestrator can't handle `tool` role messages in the react agent loop.
5. **`_screen_agent_output`** — post-screens the agent's response text with HAP and built-in detectors (`GUARDRAILS_DETECTORS_OUTPUT_SCREEN`). Gibberish excluded (false positives on menu item lists).

#### Detector notes

- **Prompt injection** — `protectai/deberta-v3-base-prompt-injection-v2` (DeBERTa, 184M params, 22 datasets). Replaced `jackhhao/jailbreak-classifier` which was poorly calibrated.
- **False positives on system prompts** — gibberish, built-in, and prompt-injection detectors all trigger on agent system prompts (the SECURITY section contains "ignore previous instructions" which triggers DeBERTa). This is why agents use regular LLMs and screening is done on isolated message text.

### NeMo Guardrails

When `NEMO_GUARDRAILS_URL` is set, a `ChatOpenAI` instance (`nemo_llm`) is created pointed at the NeMo server. NeMo uses an OpenAI-compatible `/v1/chat/completions` endpoint. Blocking is detected by checking for canned response patterns (`"I'm sorry, I can't respond to that"`, `"I can't help with that type of request"`).

The NeMo service runs on port 8000 inside its pod but is fronted by a kube-rbac-proxy (HTTPS/443). A separate internal Service (`nemo-guardrails-internal`) exposes port 8000 directly, bypassing the RBAC proxy.

#### NeMo Screening flow

1. **`_screen_nemo_input`** — sends the user's raw message to `nemo_llm`, checks response for blocked patterns.
2. **Supervisor routing** — regular LLM with structured output (no guardrails).
3. **Supervisor direct response** (route=none) — `nemo_supervisor_agent` (uses `nemo_llm`), response checked for blocked patterns.
4. **Agent nodes** — regular agents with tools (regular LLM).
5. **`_screen_nemo_output`** — sends agent response text to `nemo_llm`, checks for blocked patterns.

### "Both" Mode

When both FMS and NeMo are active, screening layers both systems sequentially — if either blocks, the message is blocked:

1. FMS `_screen_user_input` → NeMo `_screen_nemo_input`
2. Supervisor routing (regular LLM)
3. Direct response via FMS `guardrails_llm` (with detectors)
4. Agent nodes (regular LLM with tools)
5. FMS `_screen_agent_output` → NeMo `_screen_nemo_output`

### Helm chart configuration

```yaml
guardrails:
  enabled: false
  url: "http://guardrails-maas-proxy:8033/v1"

nemoGuardrails:
  enabled: false
  url: "http://nemo-guardrails-internal:8000/v1"
```
