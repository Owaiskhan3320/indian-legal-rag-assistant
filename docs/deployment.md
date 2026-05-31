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
  + demo video
  + Docker setup
  + Cloudflare live demo on request
```
