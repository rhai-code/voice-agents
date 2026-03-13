# voice-agents

A voice agent demo simulating "Pizza Palace" - ordering a pizza.

- RedHat AI Platform
- Open Source models
  - Speech to text using whisper LLM
  - Text to speech using higgs-audio LLM
  - Supervisor using llama-4-scout LLM (maas)
- Agent handoffs using Langchain + Langgraph
- Next.js web ui using websockets
- Python websocket server backend
- (Optional) MLFlow tracing

## 🏃‍♀️ Running the docs site

If you have Node installed, you can start the site with the following command from within the `top directory of the git repo` directory:

```bash
npm i docsify-cli -g
docsify serve docs/
```

## Helm Install

Deploy the voice agent stack (backend, frontend, MLflow) to OpenShift/Kubernetes:

```bash
helm upgrade --install ai-voice-agent ./ai-voice-agent/deploy/chart \
    --set backend.env.BASE_URL="<LLM_ENDPOINT_URL>/v1" \
    --set backend.env.MODEL_NAME="<LLM_MODEL_NAME>" \
    --set backend.env.TTS_URL="<TTS_ENDPOINT_URL>/v1" \
    --set backend.env.TTS_MODEL="<TTS_MODEL_NAME>" \
    --set backend.env.TTS_VOICE="belinda" \
    --set backend.env.STT_URL="<STT_ENDPOINT_URL>/v1/audio/transcriptions" \
    --set backend.env.STT_MODEL="whisper" \
    --set backend.secret.API_KEY="<YOUR_API_KEY>" \
    --set backend.secret.STT_TOKEN="<YOUR_STT_TOKEN>"
```

To disable MLflow tracing:

```bash
    --set mlflow.enabled=false
```

To point at an external MLflow instance instead of the in-cluster one:

```bash
    --set mlflow.enabled=false \
    --set backend.env.MLFLOW_TRACKING_URI="http://your-mlflow-host:5500"
```

## Install tutorial link

In an OpenShift cluster

```bash
oc apply -f odh-document.yaml
oc -n redhat-ods-applications delete pod -l app=rhods-dashboard
```
