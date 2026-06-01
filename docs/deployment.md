# Deployment

## Local Docker Deployment

The project uses Docker Compose with two services:

```text
nyaya-api  -> FastAPI backend, port 8000
nyaya-web  -> Streamlit frontend, port 8501
```

Start:

```powershell
cd C:\Project
docker compose up -d --build
```

For a clone that does not include the full local indexes, use public demo mode first:

```powershell
cd C:\Project
$env:DEMO_MODE="true"
docker compose --profile demo run --rm nyaya-demo-builder
docker compose up -d --build
```

Demo mode builds and loads only `artifacts/demo/`. It does not modify the full judgment vector DB or the full reference-law indexes.

Open:

```text
http://127.0.0.1:8501
```

Check backend:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Stop:

```powershell
docker compose down
```

## Cloudflare Tunnel Demo

After Docker is running:

```powershell
cloudflared tunnel --url http://127.0.0.1:8501
```

Cloudflare returns a temporary public URL. This link is suitable for scheduled demos, interviews, and supervisor review.

## Important Notes

- The Cloudflare quick-tunnel URL changes when the tunnel restarts.
- The link works only while the local machine, Docker containers, and tunnel process are running.
- For permanent public hosting, use a named Cloudflare Tunnel with a custom domain or move the backend to a VM.
- Large retrieval artifacts should be mounted as local volumes and should not be baked into the Docker image.

## Recommended Hiring Demo Setup

```text
GitHub repository
  + README
  + architecture docs
  + evaluation docs
  + public demo mode
  + Docker setup
  + Cloudflare live demo on request
```
