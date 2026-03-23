# Voice Agents

AI voice agent demo built on LangGraph with TrustyAI Guardrails integration.

## Project Structure

```
ai-voice-agent/
  backend/          Python WebSocket server (LangGraph agent graph)
  frontend/         React/Next.js UI
  deploy/chart/     Helm chart for OpenShift deployment
  guardrails/       Kustomize resources for TrustyAI Guardrails Orchestrator
```

## Guardrails

When `GUARDRAILS_URL` is set, the backend creates separate `ChatOpenAI` instances pointed at the orchestrator via the nginx proxy. Detectors are passed in `extra_body`. Guardrails agents use `create_react_agent` with tool-calling, same as the non-guardrails path.

1. **User input pre-screening** — the supervisor uses the regular LLM (no guardrails) for routing, so user messages are pre-screened via `_screen_user_input` with all four detectors (`GUARDRAILS_DETECTORS_INPUT_ONLY`) before routing. Only the single user message is sent to avoid false positives.
2. **Input detectors false-positive on internal messages** — agent system prompts trigger gibberish, built-in, and prompt-injection detectors (the SECURITY section contains "ignore previous instructions" which triggers DeBERTa). Agent nodes use `GUARDRAILS_DETECTORS_AGENT` (HAP only on input). User input is already pre-screened before routing.
3. **Prompt injection detector** — `protectai/deberta-v3-base-prompt-injection-v2` (DeBERTa, 184M params, 22 datasets). Replaced `jackhhao/jailbreak-classifier` which was poorly calibrated.
