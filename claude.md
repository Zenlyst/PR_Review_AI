# PR Review AI — Project Overview

## 📌 Project Summary

Fine-tune an open-source LLM on real human code review data from GitHub, then deploy it as a GitHub App that auto-reviews Pull Requests. The entire system (model serving, API, webhook handler) is packaged as a Helm chart for Kubernetes deployment.

**GitHub Repo:** https://github.com/Zenlyst/PR_Review_AI

---

## 🏗️ Architecture (Three Layers)

### Layer 1 — Model Fine-tuning
- **Base model:** CodeLlama-7B-Instruct (or Mistral-7B)
- **Method:** QLoRA (4-bit quantization + LoRA)
- **Dataset:** [`ronantakizawa/github-codereview`](https://huggingface.co/datasets/ronantakizawa/github-codereview) — 355K+ rows
- **Training scope:** Python only (filtered by `language` and `repo_language` columns)
- **Training environment:** Google Colab Pro (T4/A100 GPU)
- **Output:** ~20-30MB LoRA adapter weights

### Layer 2 — Backend Services
- **Model serving:** Direct inference via Transformers + PEFT (loads base model + LoRA adapter at startup)
- **API layer:** FastAPI — receives GitHub webhooks, runs review pipeline in background tasks
- **GitHub integration:** GitHub App with JWT auth → installation tokens → PR file fetching + review posting
- **Database:** PostgreSQL (async via SQLAlchemy + asyncpg) — stores review history
- **Flow:** PR opened/synchronized → GitHub webhook → verify HMAC signature → FastAPI background task → fetch before/after code per file → model inference → post review comments back to PR

### Layer 3 — K8s Deployment & DevOps
- **Packaging:** Helm chart (model server + API + DB)
- **CI/CD:** GitHub Actions (test → build image → deploy)
- **Ingress:** External traffic routing to API service

---

## 🔧 Tech Stack

| Category | Technologies |
|---|---|
| Language | Python |
| Fine-tuning | Hugging Face Transformers, PEFT/LoRA, bitsandbytes |
| Model Serving | Transformers + PEFT (direct inference) |
| Backend API | FastAPI, httpx, PyJWT |
| Database | PostgreSQL, SQLAlchemy (async), asyncpg |
| Config | pydantic-settings (.env) |
| Webhook Tunnel | smee.io (dev) |
| Infrastructure | Kubernetes, Helm |
| CI/CD | GitHub Actions |
| Training Env | Google Colab Pro |

---

## 📊 Dataset Details

**Source:** [`ronantakizawa/github-codereview`](https://huggingface.co/datasets/ronantakizawa/github-codereview)

| Split | Rows |
|---|---|
| Train | ~334K |
| Validation | ~10.5K |
| Test | ~11K |

### Key Columns
| Column | Description |
|---|---|
| `before_code` | Code before the change (~50 lines context) |
| `after_code` | Code after the change |
| `reviewer_comment` | Human reviewer's inline comment |
| `diff_context` | Diff format of the change |
| `file_path` | Source file path |
| `language` | Language of the **file** (37 languages) |
| `repo_language` | Primary language of the **repository** |
| `quality_score` | 0.07–1.0, higher = better review quality |
| `comment_type` | 9 categories (question, suggestion, refactor, etc.) |
| `is_negative` | `True` = clean code, no issues (negative example) |
| `repo_stars` | Star count of source repo |

### Filtering Strategy (This Project)
- **Language filter:** `language == "Python"` (file-level)
- **Quality filter:** `quality_score >= 0.5` (high-quality reviews only)
- **Code length filter:** 5–200 lines in both `before_code` and `after_code`
- **Token length filter:** remove samples exceeding 2048 tokens after formatting
- **Include both:** positive examples (real reviews) + negative examples ("No issues found")

---

## 🧠 Training Prompt Format

**Decision:** Use `before_code` + `after_code` (not `diff_context`) — provides full context like a real reviewer sees.

### Training prompt template:
```
### Instruction:
You are a senior code reviewer. Compare the before and after versions of the code below. Identify potential issues and provide improvement suggestions.

### File: {file_path}

### Before:
{before_code}

### After:
{after_code}

### Review:
{reviewer_comment}
```

### Inference prompt (Review field left empty for model to generate):
```
### Instruction:
You are a senior code reviewer. Compare the before and after versions of the code below. Identify potential issues and provide improvement suggestions.

### File: {file_path}

### Before:
{before_code}

### After:
{after_code}

### Review:

```

---

## 📅 Development Phases

### Phase 1: LoRA Fine-tuning ← v1 COMPLETE
- [x] Select dataset
- [x] Data exploration & filtering (Python only, quality + code-length filtering)
- [x] Prepare prompt templates & tokenization
- [x] Implement QLoRA training pipeline (script ready — not yet run on GPU)
- [x] Implement inference module with test examples
- [x] Run QLoRA fine-tuning on Colab Pro
- [x] Evaluate model output quality (see v1 evaluation notes below)
- [x] Save LoRA adapter to Google Drive

### Phase 1.5: Model Quality Improvements (Future)
- [ ] Train additional epochs (currently 1 epoch on 10K samples)
- [ ] Increase `TRAIN_SAMPLE_LIMIT` beyond 10K
- [ ] Try higher `QUALITY_THRESHOLD` (0.7 instead of 0.5)
- [ ] Re-evaluate on the same 3 test cases + add more

### Phase 2: Backend API ← **CURRENT** (server verified on Colab 2026-04-05)
- [x] FastAPI app with lifespan (model load on startup, DB init)
- [x] GitHub App registration (App ID: 3270311)
- [x] Webhook endpoint with HMAC-SHA256 signature verification
- [x] GitHub App JWT auth + installation token exchange
- [x] PR file fetching (Python files only) + before/after code extraction
- [x] Integration with model inference (`model_service` wraps `training.inference`)
- [x] Review pipeline (background task: fetch files → infer → post comments → save to DB)
- [x] PostgreSQL review history storage (async SQLAlchemy + asyncpg, SQLite fallback for Colab)
- [x] Config via pydantic-settings (.env), PEM key loaded from file path
- [x] Server startup verified on Colab Pro T4 (model loads, health check passes)
- [x] Webhook signature verification tested (fake payload → `{"status": "ignored"}`)
- [x] smee.io tunnel for GitHub webhook forwarding
- [ ] End-to-end test: open a real PR → receive webhook → post AI review comment

### Phase 3: K8s Deployment
- [ ] Dockerize all services
- [ ] Create Helm chart
- [ ] GitHub Actions CI/CD pipeline
- [ ] End-to-end deployment test

---

## 🔑 Key Design Decisions

1. **QLoRA over full fine-tuning** — Trainable params ~0.06% of total. Fits on Colab T4 (16GB VRAM).
2. **before/after code over diff** — Gives model full context, mimics how real reviewers read code.
3. **Python-only training** — Focused scope for v1, can expand to other languages later.
4. **Include negative examples** — Model learns when code is fine and doesn't need comments.
5. **1 epoch first** — 3 epochs was too slow for Colab (~250h ETA). Train 1 epoch, evaluate, then add more if needed.
6. **Packing enabled** — Packs multiple short samples into one sequence for better GPU utilization.
7. **Epoch-level evaluation** — Per-step eval on the full validation set was the main speed bottleneck; switched to eval once per epoch.
8. **Direct inference over vLLM/TGI** — For v1, the API loads the model directly via Transformers + PEFT. Simpler to deploy; can migrate to vLLM later if latency matters.
9. **Background task reviews** — Webhook returns 202 immediately; review pipeline runs as a FastAPI background task to avoid GitHub's 10s webhook timeout.
10. **PEM key from file path** — `.env` stores `GITHUB_PRIVATE_KEY_PATH` pointing to the `.pem` file rather than embedding key content inline. Safer and easier to rotate.

---

## 📁 Project Structure

```
pr-review-ai/
├── CLAUDE.md                 # This file — project documentation
├── training/
│   ├── __init__.py
│   ├── config.py             # All hyperparameters, paths, and prompt templates
│   ├── data_prep.py          # Dataset loading, filtering, prompt formatting
│   ├── model.py              # Model loading (4-bit QLoRA), LoRA application
│   ├── train.py              # Training orchestration + checkpoint resume logic
│   ├── inference.py          # Review generation + test examples
│   ├── setup_colab.py        # Colab environment setup
│   └── lora_finetune_pseudocode.py  # Original pseudocode (reference)
├── api/                      # Phase 2 (implemented)
│   ├── main.py               # FastAPI app — lifespan, routes (/webhook, /health, /reviews)
│   ├── config.py             # pydantic-settings config (loads .env, reads PEM from file)
│   ├── webhook.py            # HMAC-SHA256 signature verification + PR event parsing
│   ├── github_service.py     # GitHub App auth (JWT + installation tokens), PR API calls
│   ├── model_service.py      # Singleton model loader, wraps training.inference
│   ├── diff_parser.py        # Fetches before/after file content at base/head SHAs
│   ├── pipeline.py           # Review orchestration (fetch → infer → post → save)
│   ├── database.py           # Async SQLAlchemy engine + session factory
│   ├── models.py             # ORM model (Review table)
│   └── requirements.txt      # Phase 2 Python dependencies
├── helm/                     # Phase 3 (not started)
│   └── pr-review-ai/         # Helm chart
├── .github/
│   └── workflows/            # CI/CD pipelines
├── docs/
│   ├── QUICK_START.md        # Setup and run guide
│   ├── BUG_TRACKING.md       # Known issues and fixes
│   └── v1_evaluation.md      # Model evaluation results
├── 1_lora_fine_runing.ipynb  # Colab notebook — Phase 1 training
├── 2_api.ipynb               # Colab notebook — Phase 2 API server
├── .env                      # Environment config (gitignored)
├── .gitignore
├── Dockerfile
└── README.md
```

---

## 💡 Notes

- **Colab setup:** `main.ipynb` connects to Colab kernel via local IDE (Antigravity). This allows local code editing while executing on Colab GPU.
- **LoRA adapter output:** Only ~20-30MB, not the full 14GB model. Deployment loads base model + adapter.
- **Token budget:** `MAX_SEQ_LENGTH = 2048` (training). CodeLlama's context window is 16K, but samples are capped at 2048 to keep training efficient on T4.
- **Training speed tuning:** Save/eval every 200 steps (was 100). Packing=True. Eval per epoch (not per step). These changes reduced Colab ETA from ~250h to a few hours.
- **Checkpoint resume:** `train.py` auto-detects the latest checkpoint in Google Drive and resumes — handles Colab disconnects gracefully.
- **Google Drive paths:** All outputs (checkpoints, logs, adapter, dataset cache) persist to `/content/drive/MyDrive/pr-review-ai/` across sessions.
- **Local adapter copy:** The LoRA adapter is also stored locally at `./code-review-lora-adapter/` (~20MB) for API development without Colab/Drive access.
- **Environment config:** All API settings live in `.env` (gitignored). Required vars: `GITHUB_APP_ID`, `GITHUB_PRIVATE_KEY_PATH`, `GITHUB_WEBHOOK_SECRET`, `DATABASE_URL`, `ADAPTER_PATH`.
- **API startup:** `uvicorn api.main:app --reload --port 8000` — requires PostgreSQL running and `.env` configured. Model loading takes ~30-60s on first startup (downloads base model if not cached).
- **Colab dev workflow:** Edit code locally → `git push` → re-run setup cell in `2_api.ipynb` (runs `git pull` + reloads modules). Notebook cell changes must be applied manually in Colab.
- **smee.io tunnel:** Used in dev to forward GitHub webhooks to Colab. smee client runs as a background subprocess in the notebook.
- **macOS limitation:** `bitsandbytes` 4-bit quantization requires CUDA. The API server cannot load the model on macOS — use Colab Pro with T4/A100 GPU.

---

## 📊 v1 Model Evaluation (2026-04-04)

**Training config:** 1 epoch, 10K samples, `quality_score >= 0.5`, CodeLlama-7B-Instruct + QLoRA

| Test Case | Expected Issue | Model Output | Verdict |
|---|---|---|---|
| JSON→YAML migration (no error handling) | Missing `yaml.YAMLError` handling | Only suggested `import yaml` | ❌ Shallow |
| SQL query without parameterization | SQL injection via f-string | Echoed the code back, missed injection | ❌ Missed |
| Missing resource cleanup (`open()` without `with`) | File handle not closed | Correctly identified, suggested `with open()` | ✅ Good |

**Result:** 1/3 useful reviews. Acceptable as a v1 prototype — model can identify some patterns but misses critical security issues and gives shallow suggestions on others. See Phase 1.5 for improvement ideas.
