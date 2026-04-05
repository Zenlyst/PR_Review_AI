# Quick Start Guide

Get the PR Review AI server running on Google Colab Pro and receiving live GitHub webhooks.

## Prerequisites

- Google Colab Pro account (T4 or A100 GPU runtime)
- GitHub App created at https://github.com/settings/apps with:
  - **Webhook URL:** your smee.io channel URL
  - **Permissions:** Pull Requests (read & write), Contents (read)
  - **Events:** Pull request
- smee.io channel URL (get one at https://smee.io)
- LoRA adapter saved to Google Drive at `/MyDrive/pr-review-ai/code-review-lora-adapter/`
- GitHub App private key (`.pem`) saved to Google Drive at `/MyDrive/pr-review-ai/`

## 1. Open Notebook

Open `2_api.ipynb` in Google Colab. Make sure to select a **GPU runtime**:

> Runtime > Change runtime type > GPU (T4)

## 2. Run Setup Cells

Run cells in order:

| Cell | What it does |
|---|---|
| Mount Drive | Connects Google Drive at `/content/drive/` |
| Clone / Pull | Clones the repo on first run, `git pull` on subsequent runs |
| Install deps | `pip install -r api/requirements.txt` |
| Verify GPU | Confirms CUDA is available |

## 3. Configure Environment

Update the environment cell with your values:

```python
os.environ["GITHUB_APP_ID"] = "3270311"
os.environ["GITHUB_WEBHOOK_SECRET"] = "<your webhook secret>"
os.environ["GITHUB_PRIVATE_KEY_PATH"] = "/content/drive/MyDrive/pr-review-ai/<your-key>.pem"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///reviews.db"
os.environ["ADAPTER_PATH"] = "/content/drive/MyDrive/pr-review-ai/code-review-lora-adapter"
```

## 4. Smoke Test

Run the smoke test cell. Expected output:

```
App ID: 3270311
PEM path: /content/drive/MyDrive/pr-review-ai/<your-key>.pem
Adapter: /content/drive/MyDrive/pr-review-ai/code-review-lora-adapter
DB: sqlite+aiosqlite:///reviews.db
JWT signing: OK (446 chars)
```

## 5. Start smee Tunnel

Update `SMEE_URL` in the smee cell with your channel URL, then run it. This forwards GitHub webhooks to your Colab server.

```
smee client running (PID: 12345)
Forwarding: https://smee.io/abc123 → http://localhost:8000/webhook
```

## 6. Start Server

**Option A — Foreground (see logs):**

Run the `uvicorn` cell. Wait for:

```
Model loaded successfully
Application startup complete.
Uvicorn running on http://0.0.0.0:8000
```

> Note: This blocks the cell. Use Option B if you need to run other cells.

**Option B — Background (run other cells while server runs):**

Run the background server cell. It starts uvicorn as a subprocess and waits 120s for model loading.

## 7. Verify

Run the health check cell:

```json
{"status": "healthy", "model_loaded": true}
```

## 8. Test with a Real PR

1. Install the GitHub App on a test repository
2. Open a PR that changes a Python file (5-200 lines)
3. The webhook fires → smee forwards to Colab → server reviews the code → comment posted on PR

## Development Workflow

### Pushing Code Changes

1. Edit code locally (e.g., files in `api/`, `training/`)
2. `git push` from your local machine
3. Re-run the **Clone / Pull** cell in Colab — it runs `git pull` and reloads modules
4. Restart the server cell

> **Note:** Changes to the notebook itself (`2_api.ipynb`) are NOT picked up by `git pull` in an already-open notebook. Apply notebook cell changes manually in Colab.

### Environment Variables

All config is set via `os.environ` in the notebook. For local development, use a `.env` file (gitignored):

```
GITHUB_APP_ID=3270311
GITHUB_PRIVATE_KEY_PATH=./your-key.pem
GITHUB_WEBHOOK_SECRET=<secret>
DATABASE_URL=postgresql+asyncpg://user@localhost/pr_review
ADAPTER_PATH=./code-review-lora-adapter
```

### Local Development (macOS)

Everything except model inference works locally:
- Database (PostgreSQL)
- Webhook signature verification
- GitHub JWT auth
- Config loading

The model requires CUDA (4-bit quantization via bitsandbytes), so full server startup only works on Colab with a GPU.

## API Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/webhook` | GitHub webhook receiver |
| GET | `/health` | Health check (model loaded status) |
| GET | `/reviews/{owner}/{repo}` | Query review history |
| GET | `/reviews/{owner}/{repo}?pr_number=1` | Filter by PR number |
| GET | `/docs` | Interactive API docs (Swagger UI) |

## Troubleshooting

| Issue | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'api'` | Re-run the Clone/Pull cell to add project root to `sys.path` |
| `FileNotFoundError` for PEM | Check the filename in `GITHUB_PRIVATE_KEY_PATH` matches the actual file on Drive |
| `TypeError: Issuer (iss) must be a string` | `git pull` latest — fixed in `github_service.py` |
| `CUDA available: False` | Change runtime: Runtime > Change runtime type > GPU (T4) |
| Health check returns empty | Model still loading — wait 1-2 minutes and retry |
| `{"status": "ignored"}` on test webhook | Expected for non-PR events. Real PR webhooks will trigger reviews |
