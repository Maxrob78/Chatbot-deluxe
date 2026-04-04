import json
import os
import base64
import io
import asyncio
from datetime import datetime
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, Response
from pdfminer.high_level import extract_text as pdf_extract_text
import httpx
import re
from ddgs import DDGS
import trafilatura
import subprocess
import pathlib
import math
import uuid
from typing import List, Dict, Any, Optional
import logging
from collections import deque, Counter

# ─────────────────────────────────────────────
#  GLOBAL STATE — AUTO-HEALING
# ─────────────────────────────────────────────
# Garde les 5 dernières erreurs critiques en mémoire
LOG_ERROR_BUFFER = deque(maxlen=5)
# Liste des connexions SSE (Server-Sent Events) pour notifier le frontend
EVENT_QUEUES: List[asyncio.Queue] = []

class AutoHealingLogHandler(logging.Handler):
    """Handler personnalisé qui capture les erreurs critiques et les envoie au frontend."""
    def emit(self, record):
        try:
            if record.levelno >= logging.ERROR:
                msg = self.format(record)
                # On capture surtout les tracebacks s'ils existent
                traceback_data = ""
                if record.exc_info:
                    import traceback
                    traceback_data = "".join(traceback.format_exception(*record.exc_info))
                
                error_event = {
                    "type": "CRASH",
                    "timestamp": datetime.now().isoformat(),
                    "message": record.getMessage(),
                    "traceback": traceback_data or msg,
                    "level": record.levelname
                }
                LOG_ERROR_BUFFER.append(error_event)
                # Notifier tous les clients connectés
                for q in EVENT_QUEUES:
                    asyncio.create_task(q.put(error_event))
        except Exception:
            pass

# Configuration globale du logging
log_handler = AutoHealingLogHandler()
log_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(log_handler)
logging.getLogger("uvicorn.error").addHandler(log_handler)

# ─────────────────────────────────────────────
#  HELPERS — DATE
# ─────────────────────────────────────────────
def get_current_date_info() -> str:
    """Retourne la date et l'heure actuelle en français pour le contexte de l'IA."""
    now = datetime.now()
    days = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
    months = ["Janvier", "Février", "Mars", "Avril", "Mai", "Juin", "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre"]
    return f"Nous sommes actuellement le {days[now.weekday()]} {now.day} {months[now.month-1]} {now.year}. Heure : {now.strftime('%H:%M')}."

async def run_command_async(command: str, cwd: str = None) -> dict:
    """Exécute une commande shell de manière asynchrone avec protection contre l'évasion."""
    try:
        # 1. Sécurité : On force le dossier de travail à rester dans le projet
        target_cwd = pathlib.Path(cwd) if cwd else _BASE_DIR
        try:
            if not str(target_cwd.resolve()).startswith(str(_BASE_DIR.resolve())):
                return {"ok": False, "error": "Accès refusé hors du répertoire projet."}
        except: pass # fallback si resolve échoue

        # 2. Exécution
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(target_cwd)
        )
        stdout, stderr = await process.communicate()
        return {
            "ok": process.returncode == 0,
            "stdout": stdout.decode(encoding="utf-8", errors="replace"),
            "stderr": stderr.decode(encoding="utf-8", errors="replace"),
            "returncode": process.returncode
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
# Chemins absolus relatifs au dossier de main.py
_BASE_DIR    = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
HISTORY_FILE = _BASE_DIR / "history.json"
CONFIG_FILE  = _BASE_DIR / "config.json"
MEMCONFIG_FILE  = _BASE_DIR / "memconfig.json"
MEMORY_FILE  = _BASE_DIR / "memory.json"
STATS_FILE   = _BASE_DIR / "usage_stats.json"
BASE_URL        = "https://openrouter.ai/api/v1"
MAX_HISTORY     = 100
MAX_FILE_MB     = 10
REQUEST_TIMEOUT = 60

# En-têtes standards pour OpenRouter (requis pour certaines optimisations comme le caching)
OPENROUTER_HEADERS = {
    "X-Title": "Chatbot Deluxe",
    "HTTP-Referer": "http://localhost:8000"
}

app = FastAPI()

# Montage des dossiers statiques pour éviter les 404 (assets, dist, etc.)
from fastapi.staticfiles import StaticFiles
for folder in ["assets", "dist", "static"]:
    full_path = os.path.join(_BASE_DIR, folder)
    if os.path.exists(full_path):
        app.mount(f"/{folder}", StaticFiles(directory=full_path), name=folder)

@app.on_event("startup")
async def startup_event():
    """Precharge les scores Artificial Analysis au demarrage et lance le watcher."""
    try:
        cfg = load_config()
        await load_aa_scores(cfg.get("aa_api_key", ""))
        # Lancer le watcher en tâche de fond
        import asyncio
        asyncio.create_task(run_project_watcher())
        print("[STARTUP] Watcher de projet lancé.")
    except Exception as e:
        print(f"[STARTUP] Erreur startup (non bloquant): {e}")


# ─────────────────────────────────────────────
#  CONFIG HELPERS
# ─────────────────────────────────────────────
def load_config() -> dict:
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            try: return json.load(f)
            except: return {}
    return {}

def save_config(data: dict):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f)

# ─────────────────────────────────────────────
#  HISTORY HELPERS
# ─────────────────────────────────────────────
def load_history() -> list:
    if not os.path.exists(HISTORY_FILE): return []
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        try: 
            data = json.load(f)
            # Nettoyer les sessions de génération de titre accidentelles
            if isinstance(data, list):
                return [c for c in data if not str(c.get("id", "")).startswith("title-gen-")]
            return []
        except: return []

def save_history(history: list):
    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

def upsert_conversation(session_id: str, model: str, messages: list):
    if not messages: return
    first = messages[0]["content"]
    if isinstance(first, list):
        raw = " ".join(b.get("text","") for b in first if b.get("type")=="text")
    else:
        raw = str(first)
    title = (raw[:45] + "…") if len(raw) > 45 else raw

    now = datetime.now()
    history = load_history()
    for conv in history:
        if conv.get("id") == session_id:
            # Garder le titre personnalisé si l'utilisateur l'a modifié
            kept_title = conv.get("title", title)
            conv.update({
                "messages": messages,
                "model": model,
                "title": kept_title,
                "date": now.strftime("%d/%m/%Y %H:%M"),
                "bumped_at": now.isoformat(),  # pour tri par activité récente
            })
            # Remonter la conversation en tête (sauf si épinglée)
            if not conv.get("pinned"):
                history.remove(conv)
                history.append(conv)
            save_history(history)
            return
    history.append({
        "id": session_id, "title": title, "model": model, "messages": messages,
        "date": now.strftime("%d/%m/%Y %H:%M"),
        "bumped_at": now.isoformat(),
    })
    save_history(history)

# ─────────────────────────────────────────────
#  STATS HELPERS
# ─────────────────────────────────────────────
def load_stats() -> dict:
    if not os.path.exists(STATS_FILE):
        return {
            "tokens_sent": 0,
            "tokens_received": 0,
            "tokens_saved_trimmer": 0,
            "tokens_cached": 0,
            "requests_count": 0,
            "estimated_eur_saved": 0.0
        }
    with open(STATS_FILE, "r", encoding="utf-8") as f:
        try: return json.load(f)
        except: return load_stats()

def save_stats(stats: dict):
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

def update_usage_stats(sent: int, received: int, saved_trim: int = 0, cached: int = 0):
    s = load_stats()
    s["tokens_sent"] += sent
    s["tokens_received"] += received
    s["tokens_saved_trimmer"] += saved_trim
    s["tokens_cached"] += cached
    s["requests_count"] += 1
    # Estimation : ~3$ par 1M tokens -> env 2.75€ par 1M tokens
    s["estimated_eur_saved"] = (s["tokens_saved_trimmer"] + s["tokens_cached"]) / 1_000_000 * 2.75
    save_stats(s)

@app.get("/api/stats")
async def get_stats_route():
    return load_stats()

@app.post("/api/stats/reset")
async def reset_stats():
    if os.path.exists(STATS_FILE): os.remove(STATS_FILE)
    return load_stats()

# ─────────────────────────────────────────────
#  HELPERS — HISTORY
# ─────────────────────────────────────────────
def trim_history(messages: list) -> list:
    """Limite l'historique aux 25 derniers messages pour éviter de saturer le contexte."""
    if len(messages) <= 25:
        return messages
    return messages[-25:]

# ─────────────────────────────────────────────
#  BUSINESS LOGIC (Git, MCP, RAG)
# ─────────────────────────────────────────────
async def run_git(args_git: list) -> dict:
    cmd = f"git {' '.join(args_git)}"
    return await run_command_async(cmd, cwd=_BASE_DIR)

async def git_status_logic():
    if not (_BASE_DIR / ".git").exists():
        return {"ok": False, "not_repo": True}
    res = await run_git(["status", "--porcelain"])
    branch = await run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    return {
        "ok": res["ok"], 
        "status": res["stdout"], 
        "branch": branch["stdout"].strip() if branch["ok"] else "???"
    }

async def git_commit_logic(message: str):
    await run_git(["add", "."])
    return await run_git(["commit", "-m", f'"{message}"'])

def mcp_ls_logic(path: str = "."):
    if not MCP_ROOT:
        return {"error": "Aucun dossier MCP configuré"}
    try:
        target = (MCP_ROOT / path).resolve()
        if not str(target).startswith(str(MCP_ROOT)):
            return {"error": "Accès refusé hors du dossier MCP"}
        entries = []
        for entry in sorted(target.iterdir()):
            if entry.name.startswith('.'):
                continue
            entries.append({
                "name": entry.name,
                "type": "dir" if entry.is_dir() else "file",
                "size": entry.stat().st_size if entry.is_file() else None,
                "ext":  entry.suffix.lstrip('.') if entry.is_file() else None,
            })
        return {"path": str(target.relative_to(MCP_ROOT)), "entries": entries}
    except Exception as e:
        return {"error": str(e)}

def mcp_read_logic(path: str):
    if not MCP_ROOT:
        return {"error": "Aucun dossier MCP configuré"}
    try:
        target = (MCP_ROOT / path).resolve()
        if not str(target).startswith(str(MCP_ROOT)):
            return {"error": "Accès refusé"}
        if not target.is_file():
            return {"error": "Fichier introuvable"}
        text = target.read_text(encoding="utf-8", errors="replace")
        return {"path": path, "content": text, "lines": len(text.splitlines())}
    except Exception as e:
        return {"error": str(e)}

def mcp_write_logic(path: str, content: str):
    if not MCP_ROOT:
        return {"error": "Aucun dossier MCP configuré"}
    try:
        target = (MCP_ROOT / path).resolve()
        if not str(target).startswith(str(MCP_ROOT)):
            return {"error": "Accès refusé"}
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {"ok": True, "path": path, "bytes": len(content.encode())}
    except Exception as e:
        return {"error": str(e)}

# ─────────────────────────────────────────────
#  FILE HELPERS
# ─────────────────────────────────────────────
def extract_file_text(data: bytes, filename: str, content_type: str) -> str:
    """Extrait le texte de divers formats (PDF, CSV, Excel, JSON, Texte)."""
    ext = os.path.splitext(filename)[1].lower()
    try:
        # 1. TEXTE BRUT & CODE
        # Fichiers texte classiques
        if filename.lower().endswith(('.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.sql', '.yaml', '.yml', '.c', '.cpp', '.h', '.java', '.sh')):
            return data.decode('utf-8', errors='replace')
        
        # 2. PDF
        elif content_type == "application/pdf" or ext == ".pdf":
            return pdf_extract_text(io.BytesIO(data))
            
        # 2b. DOCX (Word)
        elif content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" or ext == ".docx":
            import docx
            doc = docx.Document(io.BytesIO(data))
            return "\n".join([p.text for p in doc.paragraphs])
            
        # 3. JSON
        elif content_type == "application/json" or ext == ".json":
            import json
            obj = json.loads(data)
            return json.dumps(obj, indent=2, ensure_ascii=False)
            
        # 4. CSV / EXCEL (via Pandas)
        elif ext in [".csv", ".xlsx", ".xls"]:
            import pandas as pd
            buffer = io.BytesIO(data)
            if ext == ".csv":
                # On tente de lire le CSV (virgule ou point-virgule souvent)
                try: 
                    df = pd.read_csv(buffer, sep=None, engine='python')
                except: 
                    buffer.seek(0)
                    df = pd.read_csv(buffer)
            else:
                df = pd.read_excel(buffer)
            
            # Formatage en Markdown pour que l'IA comprenne la structure
            if len(df) > 150:
                prefix = f"### [Fichier {filename} - {len(df)} lignes, affichage des 150 premières]\n\n"
                return prefix + df.head(150).to_markdown(index=False)
            return f"### [Fichier {filename}]\n\n" + df.to_markdown(index=False)

    except Exception as e:
        return f"[Erreur lecture {filename} ({ext}): {e}]"
    return ""

# Essaie de charger tiktoken, sinon on garde la variable à None
try:
    import tiktoken
    _tokenizer = tiktoken.get_encoding("cl100k_base")
except ImportError:
    _tokenizer = None

def estimate_tokens(messages: list) -> int:
    total = 0
    for m in messages:
        c = m.get("content", "")
        
        # 1. Extraire tout le texte du message
        text_content = ""
        if isinstance(c, list):
            text_content = " ".join(str(b.get("text", "")) for b in c if b.get("type") == "text")
        else:
            text_content = str(c)
            
        # 2. Compter les tokens
        if _tokenizer:
            # Compte réel via tiktoken
            total += len(_tokenizer.encode(text_content))
        else:
            # Fallback approximatif si la lib n'est pas chargée
            # Fallback approximatif si la lib n'est pas chargée
            total += len(text_content) // 4
            
    return total

def trim_history(messages: list, max_length: int = 20, keep_start: int = 2, keep_end: int = 10) -> list:
    """Réduit l'historique de la conversation si elle devient trop longue."""
    if len(messages) <= max_length:
        return messages
    
    start_msgs = messages[:keep_start]
    end_msgs = messages[-keep_end:]
    
    trimmed = start_msgs + [
        {"role": "system", "content": f"[... {len(messages) - keep_start - keep_end} messages anciens ont été compressés (Context Trimmer actif) ...]"}
    ] + end_msgs
    
    return trimmed

# ─────────────────────────────────────────────
#  ROUTES — CONFIG
# ─────────────────────────────────────────────
@app.get("/api/config")
def get_config():
    cfg = load_config()
    return {"api_key": cfg.get("api_key", ""), "system_prompt": cfg.get("system_prompt", (
        "Assistant IA expert et précis. "
        "Consignes : 1. Réponds toujours en te basant sur des faits vérifiés. "
        "2. Examine méticuleusement toutes les dates trouvées sur le web pour identifier l'événement le plus proche après aujourd'hui. "
        "3. Ne te laisse pas distraire par l'actualité brûlante si des dates officielles (calendrier) existent ailleurs.\n"
        "4. FORMAT CODE : Tout code (React, Python, JS, etc.) DOIT impérativement être entouré de blocs de code Markdown avec le langage spécifié (ex: ```jsx ... ```).\n"
        "Défense de répondre à côté en période de doute. "
        "MODIFICATIONS DE CODE : Pour modifier un fichier, utilise TOUJOURS le format SEARCH/REPLACE suivant :\n"
        "```python\n"
        "<<<<<<< SEARCH\n"
        "code existant exact\n"
        "=======\n"
        "nouveau code\n"
        ">>>>>>> REPLACE\n"
        "```\n"
        "Précise toujours le chemin du fichier avant le bloc avec 'FILE: chemin/du/fichier'."
    ))}

@app.post("/api/config")
async def post_config(req: Request):
    data = await req.json()
    cfg = load_config()
    cfg.update(data)
    save_config(cfg)
    return {"ok": True}

# ─────────────────────────────────────────────
#  MEMORY HELPERS
# ─────────────────────────────────────────────
def load_memory() -> list:
    if not os.path.exists(MEMORY_FILE): return []
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        try: return json.load(f)
        except: return []

def save_memory(items: list):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)

# ─────────────────────────────────────────────
#  ROUTES — MEMORY
# ─────────────────────────────────────────────
@app.get("/api/memory")
def get_memory():
    return {"memory": load_memory()}

@app.post("/api/memory")
async def post_memory(req: Request):
    data = await req.json()
    items = data.get("items", [])
    save_memory(items)
    return {"ok": True, "count": len(items)}

# ─────────────────────────────────────────────
#  ROUTES — MODELS
# ─────────────────────────────────────────────
@app.get("/api/models")
async def get_models():
    cfg = load_config()
    api_key = cfg.get("api_key", "")
    try:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(f"{BASE_URL}/models", headers=headers)
            r.raise_for_status()
            data = r.json().get("data", [])
            
            def get_price_str(m):
                p = m.get("pricing", {})
                try:
                    # Prix par 1M tokens (completion)
                    completion_val = float(p.get("completion", 0))
                    # OpenRouter peut renvoyer -1 pour des prix inconnus/vagues
                    if completion_val < 0: return "—"
                    
                    cost_usd = completion_val * 1_000_000
                    cost_eur = cost_usd * 0.92
                    if cost_eur == 0: return "Gratuit"
                    return f"{cost_eur:.2f}€/M" if cost_eur < 1 else f"{cost_eur:.1f}€/M"
                except: return "—"

            details = {
                m["id"]: {
                    "name": m.get("name") or m["id"].split("/")[-1],
                    "price": get_price_str(m),
                    "ctx": m.get("context_length", 0)
                } for m in data
            }
            
            models = sorted([m["id"] for m in data], key=lambda x: x.lower())
            vision_models = set(
                m["id"] for m in data
                if isinstance(m.get("architecture"), dict)
                and "image" in m["architecture"].get("input_modalities", [])
            )
            return {
                "models": models, 
                "details": details,
                "count": len(models), 
                "vision_models": list(vision_models)
            }
    except Exception as e:
        fallback = ["anthropic/claude-3-haiku", "google/gemini-1.5-pro", "openai/gpt-4o"]
        return {"models": fallback, "count": len(fallback), "vision_models": fallback, "error": str(e)}


# ---------------------------------------------
#  ROUTES - MODEL GUIDE (auto via Artificial Analysis API)
# ---------------------------------------------

# Cache des scores AA en mémoire (rechargé au démarrage du serveur)
_aa_scores_cache = {}   # slug -> {"coding": x, "intelligence": x, "reasoning": x}
_aa_cache_loaded = False

async def load_aa_scores(aa_api_key: str = ""):
    """Charge les scores depuis l'API Artificial Analysis (gratuite, pas de clé requise)."""
    global _aa_scores_cache, _aa_cache_loaded
    try:
        headers = {}
        if aa_api_key:
            headers["x-api-key"] = aa_api_key
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(
                "https://artificialanalysis.ai/api/v2/data/llms/models",
                headers=headers
            )
            if r.status_code != 200:
                return False
            data = r.json().get("data", [])
            for m in data:
                slug = m.get("slug", "")
                evals = m.get("evaluations") or {}
                if slug:
                    _aa_scores_cache[slug] = {
                        "coding":       evals.get("artificial_analysis_coding_index", 0) or 0,
                        "intelligence": evals.get("artificial_analysis_intelligence_index", 0) or 0,
                        "reasoning":    evals.get("gpqa", 0) or 0,
                        "math":         evals.get("artificial_analysis_math_index", 0) or 0,
                        "speed":        m.get("median_output_tokens_per_second", 0) or 0,
                    }
            _aa_cache_loaded = True
            print(f"[AA] {len(_aa_scores_cache)} modeles charges depuis Artificial Analysis")
            return True
    except Exception as e:
        print(f"[AA] Erreur chargement scores: {e}")
        return False

def get_aa_score(openrouter_id: str, metric: str) -> float:
    """Cherche le score AA pour un modele OpenRouter par correspondance de slug."""
    if not _aa_scores_cache:
        return 0.0
    # Extraire le nom du modèle depuis l'ID OpenRouter (ex: "anthropic/claude-opus-4-5" -> "claude-opus-4-5")
    name = openrouter_id.split("/")[-1].lower().replace(":", "-").replace("_", "-")
    # Chercher dans le cache AA par correspondance partielle sur le slug
    best_score = 0.0
    best_match_len = 0
    for slug, scores in _aa_scores_cache.items():
        slug_clean = slug.lower().replace("_", "-")
        # Match si le slug AA est contenu dans le nom OR le nom est contenu dans le slug
        if slug_clean in name or name in slug_clean or _slug_similarity(name, slug_clean) > 0.7:
            match_len = len(slug_clean)
            if match_len > best_match_len:
                best_match_len = match_len
                best_score = scores.get(metric, 0.0)
    return best_score

def _slug_similarity(a: str, b: str) -> float:
    """Similarité simple basée sur les tokens communs."""
    ta = set(a.replace("-", " ").split())
    tb = set(b.replace("-", " ").split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta), len(tb))

@app.get("/api/guide")
async def get_guide():
    import re
    global _aa_cache_loaded
    cfg = load_config()
    api_key = cfg.get("api_key", "")
    aa_api_key = cfg.get("aa_api_key", "")

    # Charger les scores AA si pas encore fait
    if not _aa_cache_loaded:
        try:
            await load_aa_scores(aa_api_key)
        except Exception as e:
            print(f"[GUIDE] AA non disponible: {e}")

    # Récupérer les modèles OpenRouter
    try:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(f"{BASE_URL}/models", headers=headers)
            r.raise_for_status()
            all_models = r.json().get("data", [])
    except Exception as e:
        return {"error": str(e), "categories": [], "aa_loaded": _aa_cache_loaded}

    all_ids = {m["id"] for m in all_models}

    def get_tags(m):
        tags = []
        arch = m.get("architecture", {})
        modalities = arch.get("input_modalities", []) if isinstance(arch, dict) else []
        pricing = m.get("pricing", {})
        ctx = m.get("context_length", 0) or 0
        mid = m.get("id", "")
        name_low = mid.lower()
        try:
            if float(pricing.get("completion", 1)) == 0 and float(pricing.get("prompt", 1)) == 0:
                tags.append("free")
        except: pass
        if ":free" in mid: tags.append("free")
        if "image" in modalities: tags.append("vision")
        if ctx >= 100000: tags.append("long")
        fast_kw = ["flash","haiku","mini","nano","lite","fast","turbo","instant","small","8b","7b","3b"]
        if any(k in name_low for k in fast_kw): tags.append("fast")
        smart_kw = ["opus","pro","large","plus","ultra","max","72b","70b","90b","405b","r1","o3","o1","sonnet","gpt-4o","gemini-1.5-pro","gemini-2","gemini-3"]
        if any(k in name_low for k in smart_kw): tags.append("smart")
        return list(set(tags)) or ["smart"]

    def format_note(m, score=None, score_label=None):
        pricing = m.get("pricing", {})
        ctx = m.get("context_length", 0) or 0
        parts = []
        if score and score > 0:
            parts.append(f"{score_label}: {score:.1f}")
        try:
            cost = float(pricing.get("completion", 0)) * 1_000_000
            if cost == 0: parts.append("Gratuit")
            elif cost < 1: parts.append(f"${cost:.2f}/M tok")
            else: parts.append(f"${cost:.1f}/M tok")
        except: pass
        if ctx >= 1_000_000: parts.append(f"{ctx//1_000_000}M ctx")
        elif ctx >= 1000: parts.append(f"{ctx//1000}K ctx")
        return " - ".join(parts)

    def make_entry(m, score=None, label=None):
        mid = m["id"]
        return {
            "id": mid,
            "name": m.get("name") or mid.split("/")[-1],
            "provider": mid.split("/")[0] if "/" in mid else "",
            "tags": get_tags(m),
            "note": format_note(m, score, label),
            "score": score or 0,
        }

    # IDs verifies directement sur openrouter.ai — mars 2026
    FALLBACK_RANKED = {
        "code": [
            "anthropic/claude-opus-4.6",
            "anthropic/claude-opus-4-5",
            "google/gemini-3.1-pro-preview",
            "openai/gpt-5.4",
            "anthropic/claude-sonnet-4.6",
            "anthropic/claude-sonnet-4-5",
            "openai/o3-mini",
            "deepseek/deepseek-r1",
            "x-ai/grok-3-beta",
            "google/gemini-2.5-pro-preview",
            "deepseek/deepseek-chat",
            "qwen/qwen-2.5-coder-32b-instruct",
            "qwen/qwq-32b",
            "openai/gpt-4o",
            "meta-llama/llama-3.3-70b-instruct",
            "mistralai/codestral-2501",
            "google/gemini-2.0-flash-001",
            "openai/gpt-4o-mini",
        ],
        "reasoning": [
            "deepseek/deepseek-r1",
            "openai/o3-mini",
            "openai/o1",
            "anthropic/claude-opus-4.6",
            "google/gemini-3.1-pro-preview",
            "anthropic/claude-opus-4-5",
            "anthropic/claude-sonnet-4.6",
            "openai/gpt-5.4",
            "qwen/qwq-32b",
            "x-ai/grok-3-beta",
            "google/gemini-2.5-pro-preview",
            "google/gemini-2.0-flash-thinking-exp",
            "openai/gpt-4o",
        ],
        "writing": [
            "anthropic/claude-opus-4.6",
            "anthropic/claude-opus-4-5",
            "google/gemini-3.1-pro-preview",
            "openai/gpt-5.4",
            "anthropic/claude-sonnet-4.6",
            "anthropic/claude-sonnet-4-5",
            "mistralai/mistral-large-2411",
            "openai/gpt-4o",
            "meta-llama/llama-3.3-70b-instruct",
            "google/gemini-2.5-pro-preview",
            "google/gemini-1.5-pro",
        ],
        "default": [
            "anthropic/claude-opus-4.6",
            "google/gemini-3.1-pro-preview",
            "openai/gpt-5.4",
            "anthropic/claude-opus-4-5",
            "anthropic/claude-sonnet-4.6",
            "anthropic/claude-sonnet-4-5",
            "deepseek/deepseek-r1",
            "x-ai/grok-3-beta",
            "google/gemini-2.5-pro-preview",
            "openai/gpt-4o",
            "mistralai/mistral-large-2411",
            "meta-llama/llama-3.3-70b-instruct",
            "qwen/qwq-32b",
            "google/gemini-2.0-flash-001",
            "openai/o3-mini",
            "openai/gpt-4o-mini",
            "deepseek/deepseek-chat",
        ],
    }

    def fallback_rank(mid: str, category: str = "default") -> int:
        import re as _re
        ranked = FALLBACK_RANKED.get(category, FALLBACK_RANKED["default"])
        # Nettoyer l ID : ignorer suffixes :free/:beta et suffixes de date -YYYYMMDD
        mid_clean = mid.lower().split(":")[0]
        mid_clean = _re.sub(r"-\d{8}$", "", mid_clean)  # ex: openai/gpt-4o-2024-11-20 -> openai/gpt-4o
        for i, prefix in enumerate(ranked):
            prefix_clean = _re.sub(r"-\d{8}$", "", prefix.lower())
            # Match exact sur l ID nettoyé
            if mid_clean == prefix_clean:
                return len(ranked) - i
        return 0

    def top_by_metric(metric, score_label, require_tag=None, keywords=None, max_n=6, category="default"):
        scored = []
        for m in all_models:
            mid = m["id"]
            tags = get_tags(m)
            if require_tag and require_tag not in tags: continue
            if keywords and not any(re.search(k, mid.lower()) for k in keywords): continue
            aa_score = get_aa_score(mid, metric)
            if aa_score > 0:
                scored.append((aa_score, m))

        if len(scored) >= 3:
            # AA disponible : trier par vrai score
            scored.sort(key=lambda x: x[0], reverse=True)
            top_n = scored[:max_n]
            return [make_entry(m, round(float(sc), 1), score_label) for sc, m in top_n]

        # Fallback : utiliser la liste curated ordonnee
        candidates = [m for m in all_models
                      if (not require_tag or require_tag in get_tags(m))
                      and (not keywords or any(re.search(k, m["id"].lower()) for k in keywords))]
        candidates.sort(key=lambda m: fallback_rank(str(m["id"]), category), reverse=True)
        # Filtrer les inconnus (score=0) sauf si pas assez de connus
        known = [m for m in candidates if fallback_rank(str(m["id"]), category) > 0]
        result_list = list(known if len(known) >= 3 else candidates)
        return [make_entry(m) for m in result_list[:max_n]]

    categories = [
        {
            "id": "code", "icon": "\u2328",
            "name": "Code & Developpement",
            "desc": "Classes par AA Coding Index — fallback: SWE-bench approx",
            "models": top_by_metric("coding", "Coding", max_n=6, category="code"),
        },
        {
            "id": "vision", "icon": "\U0001f441",
            "name": "Vision & Images",
            "desc": "Modeles supportant les images, classes par intelligence",
            "models": top_by_metric("intelligence", "Score", require_tag="vision", max_n=6, category="default"),
        },
        {
            "id": "writing", "icon": "\u270d",
            "name": "Redaction & Creativite",
            "desc": "Classes par AA Intelligence Index — fallback: liste curated",
            "models": top_by_metric("intelligence", "Score",
                keywords=["claude","mistral","gemma","llama","gpt","gemini","command","grok"],
                max_n=6, category="writing"),
        },
        {
            "id": "reasoning", "icon": "\U0001f9e0",
            "name": "Raisonnement & Analyse",
            "desc": "Classes par GPQA Diamond — fallback: liste curated",
            "models": top_by_metric("reasoning", "GPQA", max_n=6, category="reasoning"),
        },
        {
            "id": "fast", "icon": "\u26a1",
            "name": "Vitesse & Quotidien",
            "desc": "Modeles rapides detectes automatiquement",
            "models": top_by_metric("intelligence", "Score", require_tag="fast", max_n=6, category="default"),
        },
        {
            "id": "free", "icon": "\U0001f193",
            "name": "Modeles Gratuits",
            "desc": "Gratuits sur OpenRouter, classes par intelligence",
            "models": top_by_metric("intelligence", "Score", require_tag="free", max_n=6, category="default"),
        },
    ]

    return {
        "categories": categories,
        "total_models": len(all_models),
        "aa_loaded": _aa_cache_loaded,
        "aa_models_scored": len(_aa_scores_cache),
    }

@app.post("/api/guide/refresh")
async def refresh_guide():
    """Force le rechargement des scores Artificial Analysis."""
    global _aa_cache_loaded
    _aa_cache_loaded = False
    cfg = load_config()
    ok = await load_aa_scores(cfg.get("aa_api_key", ""))
    return {"ok": ok, "models_scored": len(_aa_scores_cache)}


# ─────────────────────────────────────────────
#  WEB SEARCH
# ─────────────────────────────────────────────
import re as _re
import urllib.parse

async def fetch_page_content(url: str, max_chars: int = 5000) -> str:
    """Récupère et nettoie proprement le texte d'une page web via trafilatura."""
    try:
        # 1. Sécurité : Ignorer les URLs non-HTTP ou fichiers lourds/binaires
        if not url.startswith(("http://", "https://")): return ""
        if any(url.lower().endswith(ext) for ext in [".pdf", ".jpg", ".png", ".gif", ".zip", ".mp4", ".tar", ".gz"]): return ""

        # 2. Télécharger la page via httpx (Bypass captcha / Cloudflare light)
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
            "Referer": "https://www.google.com/",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "cross-site",
        }
        # Vérification SSL activée par défaut pour la sécurité (MITM protection)
        async with httpx.AsyncClient(timeout=12, follow_redirects=True, verify=True) as client:
            try:
                r = await client.get(url, headers=headers)
            except (httpx.ConnectError, httpx.ConnectTimeout, Exception):
                # Fallback uniquement si vraiment nécessaire (ex: serveurs foot mal configurés)
                async with httpx.AsyncClient(timeout=12, follow_redirects=True, verify=False) as client_insecure:
                    r = await client_insecure.get(url, headers=headers)
            if r.status_code != 200: 
                print(f"[FETCH] HTTP {r.status_code} sur {url}")
                return ""
            html = r.text

        # 3. Extraction intelligente (Trafilatura est le meilleur pour les articles)
        content = trafilatura.extract(html, include_comments=False, include_tables=True, no_fallback=False)
        
        if not content: 
             # Fallback sur texte brut si trafilatura échoue
             # IMPORTANT: On remplace les balises par des espaces pour ne pas coller les mots (ex: <div>Equipe1</div><div>Equipe2</div>)
             text = re.sub(r'<(script|style|header|footer|nav)[^>]*>.*?</\1>', ' ', html, flags=re.S|re.I)
             text = re.sub(r'<[^>]+>', ' ', text)
             text = re.sub(r'\s+', ' ', text).strip()
             content = text
             
        # Nettoyage LLM : lignes trop courtes souvent inutiles (menus), mais on garde les scores (>10 chars)
        lines = [line.strip() for line in content.split("\n") if len(line.strip()) > 10]
        content = "\n".join(lines)
        return content[:max_chars].strip()
    except Exception as e:
        print(f"[FETCH ERROR] {url}: {e}")
        return ""


async def ddg_search(query: str, max_results: int = 5, fetch_content: bool = True) -> list[dict]:
    """Recherche DuckDuckGo robuste via l'API (version stable sans AsyncDDGS)."""
    try:
        results = []
        import asyncio
        
        # On crée une petite fonction synchrone pour la recherche pure
        def fetch_ddg():
            with DDGS() as ddgs:
                return list(ddgs.text(query, region='fr-fr', safesearch='moderate', max_results=max_results))
        
        # On l'exécute dans un thread séparé pour ne pas bloquer ton serveur rapide (FastAPI)
        ddg_results = await asyncio.to_thread(fetch_ddg)
        
        for r in ddg_results:
            results.append({
                "title": r.get("title", ""),
                "snippet": r.get("body", ""),
                "url": r.get("href", ""),
                "href": r.get("href", "")
            })

        if not results:
            return results

        # Fetcher le contenu des 4 premières pages en parallèle avec timeout court
        if fetch_content:
            top = results[:4]
            # Création de tâches asynchrones avec timeout individuel pour chaque page
            async def fetch_with_timeout(r):
                try:
                    # On réduit encore le timeout global du fetch à 4s
                    return await asyncio.wait_for(fetch_page_content(r["href"], max_chars=1200), timeout=4.0)
                except:
                    return ""

            contents = await asyncio.gather(*[fetch_with_timeout(r) for r in top])
            for i, c in enumerate(contents):
                if c:
                    results[i]["content"] = c
                    print(f"[SEARCH] Page {i+1} fetchée: {len(c)} chars")

        return results
    except Exception as e:
        print(f"[SEARCH] Erreur API DDG: {e}")
        return []


# --- REGEX PRÉ-COMPILÉES POUR LES PERFORMANCES ---

# 1. Requêtes explicites (doivent toujours passer)
RE_EXPLICIT_SEARCH = re.compile(r"\b(recherche|googler|cherche sur internet|trouve moi|search for|actualité|news|breaking|aujourd'hui|météo)\b", re.I)

# 2. Interdictions (sauf si requête explicite)
RE_NO_SEARCH = re.compile(
    r"^(salut|bonjour|hello|hi|merci|thanks|ok|oui|non|yes|no|super|cool|parfait)[\s!.]*$|"
    r"\b(code|debug|fonction|function|class|script|programme|algorithm|bug|error|fix|refactor|"
    r"python|javascript|typescript|java|rust|go|php|sql|css|html|bash|shell|"
    r"array|dict|list|string|int|float|bool|null|undefined|async|await|promise|"
    r"calcul|calculer|equation|formule|math|integral|dériver|matrice|probabilit|prouve|démontre|théorème|"
    r"écris|rédige|génère|crée|invente|imagine|raconte|traduis|résume|reformule|"
    r"poème|histoire|email|lettre|rapport|essai|article|texte|slogan|description)\b|"
    r"(```|def |class |import |var |const |let |=>|\{\s*\})",
    re.I
)

# 3. Mots-clés qui nécessitent presque toujours le web
RE_ALWAYS_SEARCH = re.compile(
    r"\b(vient de|vient d'|tout juste|récemment|dernièrement|nouvellement|stats|statistiques|"
    r"prix|coût|tarif|cours|bourse|action|crypto|bitcoin|euro|dollar|taux|combien coûte|quel est le prix|"
    r"weather|température|prévision|forecast|pluie|neige|vent|"
    r"score|match|résultat|classement|ligue|championnat|liga|tournoi|gagnant|qui a gagné|quel score|quelle équipe|"
    r"PDG|CEO|président|premier ministre|ministre|directeur|qui dirige|qui est le patron|predit|prédiction|prévois|prévoir|prediction|predictions|selection|sélection|compo|composition|buteur|marquera|buteurs|qui va marquer|"
    r"élu|élection|vote|referendum|sondage|poll|"
    r"sorti|sortie|disponible|date de sortie|quand sort|nouvelle version|mise à jour)\b",
    re.I
)

# 4. Contexte récent
RE_TECH_CONTEXT = re.compile(r"\b(version|release|lancement|produit|app|logiciel|service|startup|IA|AI)\b", re.I)



def should_search(messages: list) -> str | None:
    """
    Détecte intelligemment si une recherche web est nécessaire.
    Retourne la query optimisée ou None si pas besoin de chercher.
    """
    if not messages:
        return None
        
    last = messages[-1]
    if last.get("role") != "user":
        return None
        
    raw = last.get("content", "")
    if isinstance(raw, list):
        raw = " ".join(b.get("text", "") for b in raw if b.get("type") == "text")
        
    text = raw.strip()
    low = text.lower()

    # ── 0. GESTION DU CONTEXTE (Questions de suivi) ──
    subject = ""
    import re as _re_sub
    if len(messages) > 1:
        # On cherche le sujet dans les derniers messages
        prev_text = " ".join([str(m.get("content", "")) for m in messages[-6:]]).lower()
        # Liste de clubs étendue pour la détection
        m_equipe = _re_sub.search(r"\b(psg|om|paris|marseille|real|madrid|barça|fcb|city|liverpool|bayern|juve|milan|lyon|ol|lens|monaco|nantes|lille|chelsea|arsenal|inter|atletico|madrid|nice|manchester|united|inter|dortmund|bayern)\b", prev_text)
        if m_equipe: subject = m_equipe.group(1).upper()
        elif "bourse" in prev_text or "crypto" in prev_text: subject = "Bourse"

    # Si la question est très courte, on lui injecte le sujet
    if len(text.split()) < 5 and subject and not any(k in low for k in ["psg", "foot", "bourse"]):
        text = f"{text} {subject}"
        low = text.lower()

    # ── 1. PRIORITÉ ABSOLUE : Les demandes explicites ou sportives ──
    club_in_query = _re_sub.search(r"\b(psg|om|paris|marseille|real|madrid|barça|fcb|city|liverpool|bayern|juve|milan|lyon|ol|lens|monaco|nantes|lille|chelsea|arsenal|inter|atletico|madrid|nice|manchester|united|inter|dortmund|bayern)\b", low)
    target_team = club_in_query.group(1).upper() if club_in_query else subject

    if RE_EXPLICIT_SEARCH.search(low) or any(k in low for k in ["match", "foot", "psg", "score"]) or target_team:
        if target_team and any(k in low for k in ["match", "prochain", "calendrier", "score", "buteur", "qui va marquer", "marquera"]):
             return f"calendrier prochain match {target_team} {datetime.now().year} statistiques classement historique face-à-face"
        
        if any(k in low for k in ["match", "score", "foot"]):
             return f"match football scores {datetime.now().strftime('%d %B %Y')}"
        
        if RE_EXPLICIT_SEARCH.search(low): 
            return text

    # ── 2. ENRICHISSEMENT AUTO (Finance, News) ──
    if any(k in low for k in ["bourse", "prix", "météo", "news", "actu", "foot", "match", "psg"]):
        current_year = datetime.now().year
        if str(current_year) not in low:
            text += f" {current_year}"
        return text

    # ── 3. SUJETS DYNAMIQUES ──
    if RE_ALWAYS_SEARCH.search(low):
        return text

    # ── 4. BLOCAGE : Ce qui ne nécessite manifestement pas de recherche ──
    if RE_NO_SEARCH.search(low):
        return None

    # ── 5. QUESTIONS FACTUELLES (Plus tolérant sur la casse) ──
    words = text.split()
    if len(words) <= 20 and text.endswith("?"):
        has_recent_year = bool(re.search(r"\b(202[3-9]|203[0-9])\b", text))
        
        # On vérifie si ça commence par un mot interrogatif typique
        factual_starters = (
            "qui ", "quel ", "quelle ", "quand ", "où ", "combien ", 
            "c'est quoi", "qu'est-ce que", "pourquoi", "comment",
            "who ", "what ", "when ", "where ", "how "
        )
        is_factual = low.startswith(factual_starters)
        
        if is_factual or has_recent_year:
            return text

    # ── 6. CONTEXTE LONG ──
    if len(messages) > 2:
        recent_context = " ".join(
            (m.get("content", "") if isinstance(m.get("content"), str) else "")
            for m in messages[-3:]
        ).lower()
        
        if RE_TECH_CONTEXT.search(recent_context) and len(words) <= 15 and text.endswith("?"):
            return text

    return None

def build_search_context(results: list, query: str) -> str:
    """Formate les résultats pour injection dans le contexte."""
    if not results:
        return "INFORMATION : Aucun résultat web trouvé pour cette recherche."
    # On injecte aussi la date du jour pour que l'IA compare
    date_now = get_current_date_info()
    context = (
        f"INFORMATIONS WEB (Date actuelle : {date_now}) :\n"
        f"Tu trouveras ci-dessous les résultats de recherche les plus pertinents. "
        f"Analyse chaque source attentivement pour répondre à la requête de l'utilisateur.\n\n"
    )
    
    for i, res in enumerate(results):
        txt = res.get('content', res.get('snippet', ''))
        url = res.get('url', '')
        context += f"--- SOURCE {i+1} ({url}) ---\n"
        context += f"{txt[:4000]}\n\n"
    return context

@app.get("/api/search")
async def search_endpoint(q: str = ""):
    if not q:
        return {"results": []}
    results = await ddg_search(q)
    return {"results": results, "query": q}

# ─────────────────────────────────────────────
#  LOCAL RULES LOADER (.ai-rules, .clauderules)
# ─────────────────────────────────────────────
def get_local_rules() -> str:
    """Lit les règles spécifiques au projet si MCP_ROOT est défini."""
    if not MCP_ROOT: return ""
    import pathlib
    rules_text = ""
    for rf in [".clauderules", ".cursorrules", ".ai-rules"]:
        try:
            p = pathlib.Path(MCP_ROOT) / rf
            if p.exists():
                content = p.read_text(encoding="utf-8", errors="replace").strip()
                if content:
                    rules_text += f"\n\n🚨 RÈGLES DE DÉVELOPPEMENT SPÉCIFIQUES POUR CE PROJET ({rf}) :\n{content}"
        except: pass
    return rules_text


# ─────────────────────────────────────────────
#  DEVELOPER TOOLS LOGIC (Aider-style & Git)
# ─────────────────────────────────────────────
def validate_code_syntax(file_path: pathlib.Path) -> str | None:
    """Vérifie la syntaxe d'un fichier source et retourne l'erreur si invalide."""
    ext = file_path.suffix.lower()
    try:
        # Python
        if ext == ".py":
            import ast
            content = file_path.read_text(encoding="utf-8", errors="replace")
            ast.parse(content, filename=file_path.name)
            return None
        
        # JS/TS
        elif ext in [".js", ".jsx", ".ts", ".tsx"]:
            import subprocess
            res = subprocess.run(["node", "-c", str(file_path)], capture_output=True, text=True, timeout=5)
            if res.returncode != 0:
                err = res.stderr.strip() if res.stderr else res.stdout.strip()
                return err if err else "Erreur de syntaxe JS/TS détectée (node -c)."
            return None
            
        # PHP
        elif ext == ".php":
            import subprocess
            res = subprocess.run(["php", "-l", str(file_path)], capture_output=True, text=True, timeout=5)
            if res.returncode != 0:
                return res.stdout.strip() or "Erreur de syntaxe PHP détectée."
            return None
            
        # C / C++ (Nécessite gcc)
        elif ext in [".c", ".cpp", ".cc", ".h", ".hpp"]:
            import subprocess
            # -fsyntax-only vérifie juste la syntaxe sans générer d'objet
            compiler = "g++" if ext in [".cpp", ".cc", ".hpp"] else "gcc"
            try:
                res = subprocess.run([compiler, "-fsyntax-only", str(file_path)], capture_output=True, text=True, timeout=5)
                if res.returncode != 0:
                    return res.stderr.strip() or "Erreur de syntaxe C/C++ détectée."
            except FileNotFoundError:
                pass # Compilateur non installé, on ignore silencieusement
            return None

        # Go
        elif ext == ".go":
            import subprocess
            try:
                # go build -o nul (Windows) ou /dev/null
                null_dev = "NUL" if os.name == 'nt' else "/dev/null"
                res = subprocess.run(["go", "build", "-o", null_dev, str(file_path)], capture_output=True, text=True, timeout=5)
                if res.returncode != 0:
                    return res.stderr.strip() or "Erreur de syntaxe Go détectée."
            except FileNotFoundError:
                pass
            return None

    except SyntaxError as e:
        return f"SyntaxError ligne {e.lineno}, colonne {e.offset}: {e.msg}\nLigne suspecte : {e.text}"
    except Exception:
        return None
    return None

def extract_skeleton(file_path: pathlib.Path) -> str:
    """Extrait la structure haut niveau (classes, def, objets) d'un fichier."""
    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        lines = content.splitlines()
        skeleton = []
        ext = file_path.suffix.lower()
        import re
        
        # --- PYTHON ---
        if ext == ".py":
            class_re = re.compile(r"^\s*class\s+([a-zA-Z0-9_]+)")
            def_re = re.compile(r"^\s*(?:async\s+)?def\s+([a-zA-Z0-9_]+)\s*\(")
            for i, line in enumerate(lines, 1):
                if m := class_re.search(line):
                    skeleton.append(f"L{i}: class {m.group(1)}")
                elif m := def_re.search(line):
                    indent = len(line) - len(line.lstrip())
                    prefix = "  - " if indent > 0 else "+ "
                    skeleton.append(f"L{i}: {prefix}def {m.group(1)}")
                    
        # --- JS / TS / JSX ---
        elif ext in [".js", ".jsx", ".ts", ".tsx"]:
            class_re = re.compile(r"^\s*(?:export\s+)?class\s+([a-zA-Z0-9_]+)")
            func_re = re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+([a-zA-Z0-9_]+)")
            arrow_re = re.compile(r"^\s*(?:export\s+)?const\s+([a-zA-Z0-9_]+)\s*=\s*(?:async\s+)?(?:\([^)]*\)|[a-zA-Z0-9_]+)\s*=>")
            for i, line in enumerate(lines, 1):
                if m := class_re.search(line):
                    skeleton.append(f"L{i}: class {m.group(1)}")
                elif m := func_re.search(line):
                    skeleton.append(f"L{i}: function {m.group(1)}")
                elif m := arrow_re.search(line):
                    skeleton.append(f"L{i}: const {m.group(1)} (arrow)")
        
        # --- PHP ---
        elif ext == ".php":
            php_re = re.compile(r"^\s*(?:abstract\s+|final\s+)?(?:class|trait|interface|function)\s+([a-zA-Z0-9_]+)")
            for i, line in enumerate(lines, 1):
                if m := php_re.search(line):
                    prefix = ""
                    if "function" in line: prefix = "f: "
                    elif "class" in line: prefix = "c: "
                    skeleton.append(f"L{i}: {prefix}{m.group(1)}")

        # --- C / C++ ---
        elif ext in [".c", ".cpp", ".cc", ".h", ".hpp"]:
            # Capture rudimentaire des définitions (pas de point virgule à la fin et présence de parenthèse)
            # Match: Type Name(Args) {
            cpp_func = re.compile(r"^\s*(?:[a-zA-Z0-9_<>*:]+\s+)+([a-zA-Z0-9_]+)\s*\([^)]*\)\s*(?:const)?\s*\{?$")
            cpp_class = re.compile(r"^\s*(?:class|struct)\s+([a-zA-Z0-9_]+)\s*(?::|\{|$)")
            for i, line in enumerate(lines, 1):
                if m := cpp_class.search(line):
                    skeleton.append(f"L{i}: struct/class {m.group(1)}")
                elif m := cpp_func.search(line.strip()):
                    skeleton.append(f"L{i}: func {m.group(1)}")

        # --- JAVA ---
        elif ext == ".java":
            java_class = re.compile(r"^\s*(?:public|private|protected|static|\s)*\s*(?:class|interface|enum)\s+([a-zA-Z0-9_]+)")
            java_method = re.compile(r"^\s*(?:public|private|protected|static|final|\s)*\s*[a-zA-Z0-9_<>]+\s+([a-zA-Z0-9_]+)\s*\([^)]*\)\s*\{?$")
            for i, line in enumerate(lines, 1):
                if m := java_class.search(line):
                    skeleton.append(f"L{i}: class {m.group(1)}")
                elif (m := java_method.search(line)) and "return " not in line:
                    skeleton.append(f"L{i}:   - method {m.group(1)}")

        # --- GO ---
        elif ext == ".go":
            go_type = re.compile(r"^type\s+([a-zA-Z0-9_]+)\s+(?:struct|interface)")
            go_func = re.compile(r"^func\s+(?:\([^)]+\)\s+)?([a-zA-Z0-9_]+)\s*\(")
            for i, line in enumerate(lines, 1):
                if m := go_type.search(line):
                    skeleton.append(f"L{i}: type {m.group(1)}")
                elif m := go_func.search(line):
                    skeleton.append(f"L{i}: func {m.group(1)}")

        else:
             return f"(Extension {ext} non supportée pour l'extraction de structure)"
             
        if not skeleton:
            return "(Aucune structure détectable trouvée)"
        return "\n".join(skeleton)
    except Exception as e:
        return f"(Erreur d'extraction: {str(e)})"

def apply_aider_diff(content: str, patch: str) -> str:
    """
    Applique un patch au format SEARCH/REPLACE (Aider style) avec "Fuzzy Matching".
    Gère les différences d'indentation et les espaces de fin de ligne.
    """
    import re
    
    # 1. Extraction des blocs
    pattern = r"<<<<<<< SEARCH\n([\s\S]*?)\n=======\n([\s\S]*?)\n>>>>>>> REPLACE"
    blocks = re.findall(pattern, patch, re.DOTALL)
    if not blocks:
        return content

    new_content = content
    
    for search_text, replace_text in blocks:
        # --- STRATÉGIE 1 : Match Exact (Le plus rapide) ---
        if search_text in new_content:
            new_content = new_content.replace(search_text, replace_text, 1)
            continue

        # --- STRATÉGIE 2 : Match "Simple" (Sans espaces de fin de ligne) ---
        def strip_trailing(text):
            return "\n".join(line.rstrip() for line in text.splitlines())
            
        search_stripped = strip_trailing(search_text)
        content_lines = new_content.splitlines()
        search_lines = search_text.splitlines()
        
        found_idx = -1
        # Recherche par fenêtre glissante sur les lignes du fichier
        for i in range(len(content_lines) - len(search_lines) + 1):
            window = "\n".join(line.rstrip() for line in content_lines[i:i+len(search_lines)])
            if window == search_stripped:
                found_idx = i
                break
        
        if found_idx != -1:
            content_lines[found_idx : found_idx + len(search_lines)] = replace_text.splitlines()
            new_content = "\n".join(content_lines)
            continue

        # --- STRATÉGIE 3 : Match "Fuzzy Indentation" (Décalage de blocs) ---
        def get_indent(line):
            return len(line) - len(line.lstrip())
            
        search_lines_trim = [l.strip() for l in search_lines]
        search_content_only = "\n".join(search_lines_trim)
        
        found_idx = -1
        detected_indent_diff = 0
        
        for i in range(len(content_lines) - len(search_lines) + 1):
            window_trim = [l.strip() for l in content_lines[i:i+len(search_lines)]]
            if "\n".join(window_trim) == search_content_only:
                for j in range(len(search_lines)):
                    if search_lines[j].strip():
                        detected_indent_diff = get_indent(content_lines[i+j]) - get_indent(search_lines[j])
                        break
                found_idx = i
                break
        
        if found_idx != -1:
            replace_lines = replace_text.splitlines()
            adjusted_replace = []
            for r_line in replace_lines:
                if not r_line.strip():
                    adjusted_replace.append("")
                else:
                    target_indent = max(0, get_indent(r_line) + detected_indent_diff)
                    adjusted_replace.append(" " * target_indent + r_line.lstrip())
            
            content_lines[found_idx : found_idx + len(search_lines)] = adjusted_replace
            new_content = "\n".join(content_lines)
            continue

        # --- ÉCHEC ---
        context_preview = search_text.strip()[:100] + "..." if len(search_text) > 100 else search_text.strip()
        raise Exception(f"Bloc SEARCH introuvable dans le fichier.\nExtrait cherché : '{context_preview}'")
        
    return new_content


# ─────────────────────────────────────────────
#  TRANSACTION MULTI-FICHIERS ATOMIQUE
# ─────────────────────────────────────────────
async def execute_transaction(changes: list, commit_message: str, run_tests_cmd: str = "") -> dict:
    """
    Applique une liste de changements multi-fichiers de manière atomique.
    En cas d'erreur (syntaxe ou tests), effectue un rollback Git automatique.

    Args:
        changes: Liste de dicts avec 'path' et soit 'diff' (SEARCH/REPLACE),
                 soit 'content' (écriture complète).
        commit_message: Message du commit Git si tout réussit.
        run_tests_cmd: Commande de tests optionnelle à exécuter avant le commit.

    Returns:
        dict avec 'ok' (bool), 'committed' (bool), 'results' (log par fichier),
        'error' (si échec), 'rollback' (bool).
    """
    if not MCP_ROOT:
        return {"ok": False, "error": "Aucun dossier MCP configuré."}
    if not changes:
        return {"ok": False, "error": "Aucun changement fourni."}

    results = []
    rollback_needed = False
    original_contents: dict[str, str] = {}  # Sauvegarde en mémoire pour rollback

    # --- PHASE 1 : SNAPSHOT (Sauvegarde en mémoire des fichiers originaux) ---
    for change in changes:
        path = change.get("path", "")
        if not path:
            results.append({"ok": False, "path": "?", "error": "Chemin manquant."})
            rollback_needed = True
            break
        target = (MCP_ROOT / path).resolve()
        if not str(target).startswith(str(MCP_ROOT)):
            results.append({"ok": False, "path": path, "error": "Accès refusé hors du dossier MCP."})
            rollback_needed = True
            break
        # Lire le contenu actuel seulement si le fichier existe (pas pour les nouveaux fichiers)
        if target.is_file():
            try:
                original_contents[path] = target.read_text(encoding="utf-8")
            except Exception as e:
                results.append({"ok": False, "path": path, "error": f"Lecture impossible : {e}"})
                rollback_needed = True
                break
        else:
            original_contents[path] = None  # Nouveau fichier

    if rollback_needed:
        return {"ok": False, "results": results, "rollback": False, "error": "Échec lors de la phase de snapshot."}

    # --- PHASE 2 : APPLICATION (Tous les changements en mémoire d'abord) ---
    new_contents: dict[str, str] = {}
    for change in changes:
        path = change.get("path", "")
        target = (MCP_ROOT / path).resolve()
        diff = change.get("diff", "")
        content_full = change.get("content", "")

        try:
            if diff:
                # Mode SEARCH/REPLACE
                if original_contents.get(path) is None:
                    raise Exception(f"Impossible d'appliquer un diff sur un fichier inexistant : {path}")
                new_text = apply_aider_diff(original_contents[path], diff)
            elif content_full is not None and content_full != "":
                # Mode écriture complète
                new_text = content_full
            else:
                raise Exception("Ni 'diff' ni 'content' fourni pour ce changement.")

            new_contents[path] = new_text
            results.append({"ok": True, "path": path, "action": "diff" if diff else "write"})
        except Exception as e:
            results.append({"ok": False, "path": path, "error": str(e)})
            rollback_needed = True
            break

    if rollback_needed:
        return {"ok": False, "results": results, "rollback": False,
                "error": "Échec lors de la phase d'application des diffs (aucun fichier modifié sur disque)."}

    # --- PHASE 3 : ÉCRITURE SUR DISQUE ---
    written_paths = []
    for path, new_text in new_contents.items():
        target = (MCP_ROOT / path).resolve()
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(new_text, encoding="utf-8")
            written_paths.append(path)
        except Exception as e:
            # Mettre à jour le résultat de ce fichier
            for r in results:
                if r.get("path") == path:
                    r["ok"] = False
                    r["error"] = f"Erreur écriture disque : {e}"
            rollback_needed = True
            break

    # --- PHASE 4 : VALIDATION SYNTAXIQUE ---
    if not rollback_needed:
        for path in written_paths:
            target = (MCP_ROOT / path).resolve()
            syntax_error = validate_code_syntax(target)
            if syntax_error:
                for r in results:
                    if r.get("path") == path:
                        r["ok"] = False
                        r["syntax_error"] = syntax_error
                rollback_needed = True
                break

    # --- PHASE 5 : TESTS (optionnel) ---
    if not rollback_needed and run_tests_cmd:
        res_test = await run_command_async(run_tests_cmd, cwd=MCP_ROOT)
        if not res_test["ok"]:
            rollback_needed = True
            return {
                "ok": False,
                "results": results,
                "rollback": True,
                "error": f"Tests échoués — rollback déclenché.\n{res_test['stdout']}\n{res_test['stderr']}",
                "test_output": (res_test['stdout'] + res_test['stderr']).strip()
            }

    # --- ROLLBACK (si besoin) ---
    if rollback_needed:
        restored = []
        for path, original_text in original_contents.items():
            target = (MCP_ROOT / path).resolve()
            try:
                if original_text is None:
                    # Fichier créé par la transaction → on l'efface
                    if target.exists():
                        target.unlink()
                    restored.append(f"{path} (supprimé — nouveau fichier)")
                else:
                    target.write_text(original_text, encoding="utf-8")
                    restored.append(path)
            except Exception as e:
                restored.append(f"{path} (ERREUR RESTORE: {e})")

        syntax_errors = [r.get("syntax_error") for r in results if r.get("syntax_error")]
        error_detail = "; ".join(r.get("error", "") for r in results if not r.get("ok") and r.get("error"))
        return {
            "ok": False,
            "results": results,
            "rollback": True,
            "restored_files": restored,
            "error": f"Transaction annulée — rollback effectué. Détail : {error_detail}",
            "syntax_errors": syntax_errors,
        }

    # --- PHASE 6 : COMMIT GIT ATOMIQUE ---
    committed = False
    commit_output = ""
    is_git_repo = (MCP_ROOT / ".git").exists()
    if is_git_repo:
        # git add de TOUS les fichiers modifiés
        files_to_add = " ".join(f'"{p}"' for p in new_contents.keys())
        await run_command_async(f"git add {files_to_add}", cwd=MCP_ROOT)
        res_c = await run_command_async(f'git commit -m "{commit_message}"', cwd=MCP_ROOT)
        if res_c["ok"]:
            committed = True
            commit_output = res_c["stdout"].strip()
        else:
            commit_output = res_c["stderr"].strip()

    return {
        "ok": True,
        "results": results,
        "rollback": False,
        "committed": committed,
        "commit_message": commit_message,
        "commit_output": commit_output,
        "files_changed": len(new_contents),
    }

AGENT_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Recherche des informations récentes sur le web via DuckDuckGo (snippets).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "La requête de recherche"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_url",
            "description": "Lit et extrait TOUT le texte d'une page web (URL). INDISPENSABLE pour résumer des articles, lire de la documentation technique ou analyser le contenu d'un lien fourni par l'utilisateur.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "L'URL valide à parcourir (commençant par http)."}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Lit le contenu d'un fichier du projet (nécessite MCP configuré).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Chemin relatif du fichier depuis la racine MCP"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Écrit ou modifie un fichier du projet (nécessite MCP configuré).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path":    {"type": "string", "description": "Chemin relatif du fichier"},
                    "content": {"type": "string", "description": "Contenu complet à écrire"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "Liste les fichiers d'un dossier du projet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Chemin relatif du dossier (. pour la racine)", "default": "."}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_project",
            "description": "Recherche un texte ou une fonction dans TOUS les fichiers du projet (Mini-RAG).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Le texte à rechercher"},
                    "file_extension": {"type": "string", "description": "Optionnel: limiter à une extension (ex: .py, .html)"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "map_project",
            "description": "Extrait la structure architecturale (Sitemap des classes/fonctions) d'un fichier ou de tous les fichiers d'un dossier spécifié. Idéal pour comprendre un gros fichier sans le lire en entier.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Chemin relatif du fichier ou dossier (ex: 'main.py' ou 'src/'). Si vide, cible la racine MCP.", "default": "."}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Exécute du code Python côté serveur et retourne la sortie.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Code Python à exécuter"}
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Exécute une commande shell (terminal) sur le serveur et retourne la sortie.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "La commande shell à exécuter (ex: 'ls', 'pip install', 'node script.js')"}
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "apply_diff",
            "description": "Applique des modifications partielles à un fichier via des blocs SEARCH/REPLACE (Aider style). Plus rapide et sûr que write_file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Chemin du fichier"},
                    "diff": {"type": "string", "description": "Les blocs <<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE"}
                },
                "required": ["path", "diff"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_git",
            "description": "Exécute une commande Git dans le projet (status, add, commit, branch, diff, log).",
            "parameters": {
                "type": "object",
                "properties": {
                    "args": {"type": "string", "description": "Arguments de la commande git (ex: 'status', 'commit -m \"message\"')"}
                },
                "required": ["args"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "rag_index",
            "description": "Indexe les fichiers du projet pour permettre une recherche sémantique plus tard. Peut indexer tout le projet ou une liste spécifique.",
            "parameters": {
                "type": "object",
                "properties": {
                    "all": {"type": "boolean", "description": "Si true, indexe tout le dossier MCP racine."},
                    "paths": {"type": "array", "items": {"type": "string"}, "description": "Liste optionnelle de chemins spécifiques à indexer."}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": "Effectue une recherche 'intelligente' (RAG) dans les fichiers précédemment indexés.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "La question ou le code à chercher"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Lance des tests unitaires ou un script pour valider le code. Utile pour implémenter un correctif parfait (TDD loop).",
            "parameters": {
                "type": "object",
                "properties": {
                    "test_file": {"type": "string", "description": "Le fichier de test/script à exécuter (ex: test.py, main.py)"},
                    "command": {"type": "string", "description": "Optionnel: La commande spécifique (ex: 'pytest')" }
                },
                "required": ["test_file"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": "Génère une image à partir d'une description textuelle en anglais (prompt) et l'affiche à l'utilisateur.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Description ultra-détaillée de l'image (en anglais de préférence pour une meilleure qualité)."}
                },
                "required": ["prompt"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "apply_multi_diff",
            "description": (
                "Applique des modifications sur PLUSIEURS fichiers en une seule transaction atomique. "
                "Si un seul fichier échoue (syntaxe invalide ou tests échoués), TOUS les fichiers sont "
                "restaurés automatiquement (rollback). Si tout réussit, un commit Git est créé. "
                "UTILISER EN PRIORITÉ pour toute refactorisation impliquant 2 fichiers ou plus."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "changes": {
                        "type": "array",
                        "description": "Liste des modifications à appliquer.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string", "description": "Chemin relatif du fichier depuis la racine MCP."},
                                "diff":    {"type": "string", "description": "Blocs SEARCH/REPLACE à appliquer (format Aider). Utiliser si le fichier existe déjà."},
                                "content": {"type": "string", "description": "Contenu complet à écrire (pour les nouveaux fichiers ou réécriture totale)."}
                            },
                            "required": ["path"]
                        }
                    },
                    "commit_message": {
                        "type": "string",
                        "description": "Message du commit Git (format conventionnel recommandé : 'feat: ...', 'fix: ...', 'refactor: ...')."
                    },
                    "run_tests": {
                        "type": "string",
                        "description": "Optionnel: commande de tests à lancer avant le commit (ex: 'pytest', 'npm test'). La transaction est annulée si les tests échouent."
                    }
                },
                "required": ["changes", "commit_message"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "launch_app",
            "description": "Lance une application ou un jeu local. Cherche automatiquement dans le Menu Démarrer et les dossiers de programmes si le nom est fourni (ex: 'Figma', 'Chrome', 'CapCut'). Supporte aussi les chemins (.exe) et les raccourcis (.lnk).",
            "parameters": {
                "type": "object",
                "properties": {
                    "app_name": {"type": "string", "description": "Nom de l'application ou chemin vers l'exécutable."}
                },
                "required": ["app_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "terminate_app",
            "description": "Ferme (arrête) un programme ou une application locale (ex: 'chrome', 'discord').",
            "parameters": {
                "type": "object",
                "properties": {
                    "app_name": {"type": "string", "description": "Nom de l'application à arrêter."}
                },
                "required": ["app_name"]
            }
        }
    }
]

async def execute_tool(name: str, args: dict) -> str:
    """Exécute un outil et retourne le résultat sous forme de string."""
    try:
        if name == "web_search":
            query = args.get("query","")
            print(f"[AGENT TOOL] web_search query: '{query}'")
            results = await ddg_search(query, max_results=4, fetch_content=True)
            if not results: return "Aucun résultat trouvé."
            return build_search_context(results, query)

        elif name == "terminate_app":
            app_raw = args.get("app_name","").lower().strip()
            
            # --- PROTECTIONS SYSTÈME ---
            # explorer.exe est le shell Windows (Barre des tâches + Bureau). 
            # Le fermer avec /T tue aussi les terminaux ouverts, y compris ce serveur s'il a été lancé depuis l'interface.
            if app_raw in ["explorer", "explorateur", "shell", "bureau", "taskbar"]:
                return "⚠️ Sécurité : Je ne peux pas fermer l'Explorateur Windows car cela ferait disparaître ton bureau et pourrait arrêter mon propre processus."

            # Mappage des processus connus
            PROCS = {
                "chrome": "chrome.exe", "discord": "Discord.exe", 
                "steam": "steam.exe", "spotify": "Spotify.exe", 
                "vscode": "Code.exe", "code": "Code.exe",
                "terminal": "WindowsTerminal.exe", "cmd": "cmd.exe", "powershell": "powershell.exe"
            }
            
            target = PROCS.get(app_raw, app_raw)
            if not target.endswith(".exe"): target += ".exe"
            
            print(f"[AGENT TOOL] terminate_app check: {target}")
            
            # --- SCAN DES PROCESSUS ACTIFS (Nouveau) ---
            # Si le premier essai échoue, on cherche dans la liste réelle
            import subprocess
            try:
                # On récupère la liste des processus (Lecture seule)
                task_list_proc = await run_command_async("tasklist /NH /FO CSV")
                if task_list_proc.get("ok") and task_list_proc.get("stdout"):
                    rows = task_list_proc["stdout"].splitlines()
                    for row in rows:
                        # Format CSV: "Image Name","PID","Session Name","Session#","Mem Usage"
                        # ex: "Discord.exe","1234","Console","1","50 000 K"
                        p_name = row.split('","')[0].replace('"', '')
                        p_low = p_name.lower()
                        # Si le nom demandé est contenu dans le processus (ex: "discord" match "Discord.exe")
                        if app_raw in p_low:
                            target = p_name
                            print(f"[AGENT TOOL] terminate_app found match via scan: {target}")
                            break
            except Exception as e:
                print(f"[DEBUG] terminate_app scan error: {e}")

            # Sécurité finale sur le nom de processus résolu
            CRITICAL_PROCS = ["explorer.exe", "taskhostw.exe", "svchost.exe", "winlogon.exe", "csrss.exe", "services.exe", "lsass.exe", "dwm.exe", "python.exe", "pythonw.exe"]
            if target.lower() in CRITICAL_PROCS:
                return f"⚠️ Bloqué : '{target}' est un processus système critique indispensable à Windows."

            # Utilisation de TASKKILL sur Windows
            cmd = f'taskkill /F /IM "{target}" /T'
            res = await run_command_async(cmd)
            
            if res.get("ok"):
                return f"✅ '{app_raw}' (processus {target}) a été arrêté avec succès."
            else:
                return f"⚠️ Impossible d'arrêter '{app_raw}'. Il n'est peut-être pas ouvert ou le nom de processus '{target}' est incorrect."

        elif name == "read_url":
            url = args.get("url")
            if not url: return "Erreur : URL manquante."
            # Appel asynchrone à mcp_fetch_url (qui est défini plus bas)
            res = await mcp_fetch_url(url)
            if hasattr(res, "body"): # Si c'est un JSONResponse
                import json as _json
                data = _json.loads(res.body.decode())
                return data.get("content") or data.get("error") or "Contenu vide."
            return res.get("content") if isinstance(res, dict) else str(res)

        elif name == "read_file":
            if not MCP_ROOT:
                return "Erreur : aucun dossier MCP configuré."
            import pathlib
            path = args.get("path","")
            target = (MCP_ROOT / path).resolve()
            if not str(target).startswith(str(MCP_ROOT)):
                return "Erreur : accès refusé hors du dossier MCP."
            if not target.is_file():
                return f"Erreur : fichier introuvable : {path}"
            if target.stat().st_size > 10 * 1024 * 1024:
                return "Erreur : fichier trop grand (max 10MB)."
            
            # Utiliser extract_file_text pour supporter PDF, DOCX, CSV, etc.
            content = extract_file_text(target.read_bytes(), target.name, "")
            return content if content else "(fichier vide ou format non supporté)"

        elif name == "generate_image":
            prompt = args.get("prompt", "")
            if not prompt: return "Erreur : prompt vide."
            # Utilisation de Pollinations
            import urllib.parse
            import random
            
            # On ne rajoute plus de mots-clés si le prompt est déjà riche
            enhanced = prompt[:800] 
            model = "flux" 
            seed = random.randint(0, 999999)
            
            p_uri = urllib.parse.quote(enhanced)
            # Pas besoin de clé ici, Pollinations est ouvert
            image_url = f"/api/proxy_image?prompt={p_uri}&seed={seed}&model={model}"
            
            return f"IMAGE_GENERATED: {image_url}\n"

        elif name == "write_file":
            if not MCP_ROOT:
                return "Erreur : aucun dossier MCP configuré."
            import pathlib
            path    = args.get("path","")
            content = args.get("content","")
            target  = (MCP_ROOT / path).resolve()
            if not str(target).startswith(str(MCP_ROOT)):
                return "Erreur : accès refusé."
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            
            # Validation automatique
            base_msg = f"Fichier écrit avec succès : {path} ({len(content.encode())} bytes)."
            err = validate_code_syntax(target)
            if err:
                return f"{base_msg}\n\nATTENTION : Une erreur de syntaxe a été détectée dans le code sauvegardé.\nCorrige-la immédiatement :\n{err}"
            return base_msg

        elif name == "list_files":
            if not MCP_ROOT:
                return "Erreur : aucun dossier MCP configuré."
            import pathlib
            path   = args.get("path",".")
            target = (MCP_ROOT / path).resolve()
            if not str(target).startswith(str(MCP_ROOT)):
                return "Erreur : accès refusé."
            entries = []
            for e in sorted(target.iterdir()):
                if not e.name.startswith('.'):
                    prefix = "📁" if e.is_dir() else "📄"
                    size   = f" ({e.stat().st_size} bytes)" if e.is_file() else ""
                    entries.append(f"{prefix} {e.name}{size}")
            return "\n".join(entries) if entries else "(dossier vide)"
        elif name == "run_python":
            code = args.get("code","")
            import io, contextlib, ast
            # Analyse statique intelligente (on bloque les techniques d'évasion, pas les outils)
            try:
                tree = ast.parse(code)
                for node in ast.walk(tree):
                    # 1. Bloquer l'évasion par les attributs privés (ex: __subclasses__)
                    if isinstance(node, ast.Attribute):
                        if node.attr.startswith("__") and node.attr != "__name__":
                            return f"Sécurité : L'accès aux attributs internes ('{node.attr}') est interdit."
                    
                    # 2. Bloquer les fonctions de méta-programmation dangereuses
                    elif isinstance(node, ast.Call):
                        fn_id = ""
                        if isinstance(node.func, ast.Name): fn_id = node.func.id
                        elif isinstance(node.func, ast.Attribute): fn_id = node.func.attr
                        
                        if fn_id in ['eval', 'exec', 'getattr', 'setattr', 'compile', '__import__']:
                            return f"Sécurité : L'utilisation de '{fn_id}' est bloquée pour éviter toute évasion de contexte. Utilise des fonctions standards."
            except SyntaxError as e:
                return f"Erreur de syntaxe : {e}"

            # Exécution
            buf = io.StringIO()
            try:
                safe_globals = {"__builtins__": __builtins__}
                safe_locals = {}
                with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                    exec(compile(code, "<agent>", "exec"), safe_globals, safe_locals)
                output = buf.getvalue()
                return output if output else "(pas de sortie)"
            except Exception as e:
                return f"Erreur Python : {type(e).__name__}: {e}"  

        elif name == "apply_diff":
            if not MCP_ROOT: return "Erreur : aucun dossier MCP configuré."
            path = args.get("path","")
            diff = args.get("diff","")
            target = (MCP_ROOT / path).resolve()
            if not target.is_file(): return f"Erreur : fichier introuvable : {path}"
            
            try:
                old_text = target.read_text(encoding="utf-8")
                new_text = apply_aider_diff(old_text, diff)
                target.write_text(new_text, encoding="utf-8")
                
                # Validation automatique
                base_msg = f"Fichier {path} modifié avec succès (Diff appliqué)."
                err = validate_code_syntax(target)
                if err:
                    return f"{base_msg}\n\nATTENTION : L'application du diff a introduit une ERREUR DE SYNTAXE.\nCorrige-la immédiatement :\n{err}"
                return base_msg
            except Exception as e:
                return f"Erreur application Diff : {str(e)}"

        elif name == "run_git":
            git_args = args.get("args","status")
            # Utilisation du moteur sécurisé
            res = await run_command_async(f"git {git_args}", cwd=str(MCP_ROOT if MCP_ROOT else _BASE_DIR))
            output = (res.get("stdout", "") + "\n" + res.get("stderr", "")).strip()
            if not res.get("ok"):
                return f"Erreur Git : {res.get('error', '')}\n{output}"
            return output or "(commande exécutée, pas de sortie)"
                        
        elif name == "run_tests":
            if not MCP_ROOT: return "Erreur : aucun dossier MCP configuré."
            test_file = args.get("test_file","")
            command = args.get("command","")
            
            target = (MCP_ROOT / test_file).resolve()
            if not target.is_file(): return f"Erreur : fichier de test introuvable : {test_file}"
            
            ext = target.suffix.lower()
            if not command:
                if ext == ".py": command = f"python -m pytest {test_file} -v" if target.parent.joinpath("pytest.ini").exists() or target.name.startswith("test_") else f"python {test_file}"
                elif ext in [".js", ".ts", ".jsx", ".tsx"]: command = f"npx jest {test_file}" if "test" in target.name else f"node {test_file}"
                elif ext == ".go": command = f"go test {test_file}"
                elif ext == ".php": command = f"phpunit {test_file}" if "test" in target.name else f"php {test_file}"
                else: return f"Impossible de déduire la commande de test pour cette extension ({ext}). Indiquez la commande explicitement."
            
            try:
                # Utilisation du moteur sécurisé unifié (limit au projet + timeout)
                res = await run_command_async(command, cwd=str(MCP_ROOT if MCP_ROOT else _BASE_DIR))
                out = res.get("stdout", "")
                err = res.get("stderr", "")
                result = out + "\n" + err
                
                if res.get("ok"):
                    return f"Tests réussis ({command}) :\n{result.strip()}"
                else:
                    return f"Les tests ont ÉCHOUÉ (code {res.get('returncode')}) :\n{result.strip()}\n-> Analyse l'erreur et lance à nouveau les outils de modification pour corriger le tir."
            except subprocess.TimeoutExpired:
                 return f"Erreur : Délai dépassé (timeout 30s) pour {command}"
            except Exception as e:
                return f"Erreur lors de l'exécution des tests : {str(e)}"
                
        elif name == "search_project":
            if not MCP_ROOT: return "Erreur : aucun dossier MCP configuré."
            query = str(args.get("query","")).lower()
            ext   = str(args.get("file_extension",""))
            if not query: return "Erreur : requête vide."
            
            matches = []
            import pathlib
            # Utilisation de rglob pour la récursion
            for p in MCP_ROOT.rglob("*"):
                if p.is_file():
                    if ext and not p.name.endswith(ext): continue
                    # Ignorer dossiers techniques habituels
                    parts_low = [pt.lower() for pt in p.parts]
                    if any(pt.startswith('.') for pt in p.parts) or "node_modules" in parts_low or "__pycache__" in parts_low:
                        continue
                    try:
                        content = p.read_text(encoding="utf-8", errors="ignore")
                        if query in content.lower():
                            lines = content.splitlines()
                            for i, line in enumerate(lines, 1):
                                if query in line.lower():
                                    matches.append(f"[{p.relative_to(MCP_ROOT)}] Ligne {i}: {line.strip()[:120]}")
                                    if len(matches) >= 30: break
                    except: pass
                if len(matches) >= 30: break
            
            return "\n".join(matches) if matches else "Aucune occurrence trouvée."

        elif name == "map_project":
            if not MCP_ROOT: return "Erreur : aucun dossier MCP configuré."
            path_arg = args.get("path", ".")
            import pathlib
            target = (MCP_ROOT / path_arg).resolve()
            if not str(target).startswith(str(MCP_ROOT)):
                return "Erreur : accès refusé hors du dossier MCP."
            
            if target.is_file():
                res = f"--- Structure de {target.relative_to(MCP_ROOT)} ---\n"
                res += extract_skeleton(target)
                return res
            elif target.is_dir():
                output = []
                for p in target.rglob("*"):
                    if p.is_file():
                        ext = p.suffix.lower()
                        if ext in [".py", ".js", ".ts", ".jsx", ".tsx"]:
                            parts_low = [pt.lower() for pt in p.parts]
                            if any(pt.startswith('.') for pt in p.parts) or "node_modules" in parts_low or "__pycache__" in parts_low:
                                continue
                            sk = extract_skeleton(p)
                            if not sk.startswith("("):
                                output.append(f"--- Fichier: {p.relative_to(MCP_ROOT)} ---")
                                output.append(sk)
                                output.append("")
                return "\n".join(output) if output else "(Aucun fichier de code parsable trouvé)"
            else:
                return "Chemin invalide."

        elif name == "run_shell":
            command = args.get("command", "")
            if not command: return "Erreur : commande vide."
            # Utilisation du moteur sécurisé
            res = await run_command_async(command, cwd=str(MCP_ROOT if MCP_ROOT else _BASE_DIR))
            output = res.get("stdout", "")
            if res.get("stderr"):
                output += f"\n[stderr]\n{res['stderr']}"
            if not res.get("ok"):
                return f"Erreur Shell : {res.get('error', '')}\n{output}"
            return output if output else "(commande exécutée sans sortie console)"

        elif name == "rag_index":
            is_all = args.get("all", False)
            paths = args.get("paths", [])
            if is_all:
                if not MCP_ROOT: return "Erreur : aucun dossier MCP configuré pour indexer tout le projet."
            if not query: return "Erreur : requête vide."
            res = rag_search_internal(query)
            if not res or not res.get("results"):
                return "Aucun résultat pertinent trouvé dans l'index RAG. As-tu indexé le projet (rag_index) ?"
            
            output = "[Résultats RAG]\n"
            for r in res["results"]:
                if r["score"] > 0:
                    output += f"--- FICHIER: {r['path']} (score: {r['score']:.2f}) ---\n{r['text']}\n"
            return output

        elif name == "apply_multi_diff":
            changes        = args.get("changes", [])
            commit_message = args.get("commit_message", "chore: modifications automatiques via agent")
            run_tests_cmd  = args.get("run_tests", "")

            if not changes:
                return "Erreur : la liste 'changes' est vide."

            result = await execute_transaction(changes, commit_message, run_tests_cmd)

            # Formatter un rapport lisible pour l'IA
            lines = []
            status = "✅ TRANSACTION RÉUSSIE" if result["ok"] else "❌ TRANSACTION ANNULÉE (rollback)"
            lines.append(status)
            lines.append(f"Fichiers traités : {result.get('files_changed', len(changes))}")

            if result.get("committed"):
                lines.append(f"✅ Commit Git créé : '{result['commit_message']}'")
                lines.append(f"   {result.get('commit_output', '')}")
            elif result.get("ok") and not result.get("committed"):
                lines.append("ℹ️ Pas de repo Git détecté — fichiers écrits sans commit.")

            lines.append("")
            lines.append("Détail par fichier :")
            for r in result.get("results", []):
                icon = "✅" if r.get("ok") else "❌"
                action = r.get("action", "?")
                path   = r.get("path", "?")
                err    = r.get("error", r.get("syntax_error", ""))
                lines.append(f"  {icon} [{action}] {path}" + (f" — {err}" if err else ""))

            if result.get("rollback"):
                lines.append("")
                lines.append(f"⚠️ Cause du rollback : {result.get('error', '')}")
                restored = result.get("restored_files", [])
                if restored:
                    lines.append(f"   Fichiers restaurés : {', '.join(restored)}")
                if result.get("syntax_errors"):
                    lines.append("   Erreurs syntaxe détectées :")
                    for se in result["syntax_errors"]:
                        lines.append(f"   → {se}")
                lines.append("")
                lines.append("ACTION REQUISE : Analyse les erreurs ci-dessus et corrige ta stratégie avant de relancer apply_multi_diff.")

            return "\n".join(lines)

        elif name == "launch_app":
            app_raw = args.get("app_name", "").strip()
            if not app_raw: return "Erreur : nom d'application ou URL manquant."
            
            # --- PRIORITÉ AUX URLS ---
            # Si l'entrée contient 'http', on extrait l'URL et on l'ouvre directement
            import re
            url_match = re.search(r'(https?://\S+)', app_raw)
            if url_match:
                url = url_match.group(1)
                import webbrowser
                webbrowser.open(url)
                print(f"[AGENT TOOL] launch_app (Direct URL mode): {url}")
                return f"✅ URL ouverte avec succès : {url}"

            # 1. Mappage des jeux/apps connus (prioritaire)
            app_raw = app_raw.lower()
            import os, shutil
            APPS = {
                "fc26": "steam://rungameid/3405690",
                "fifa": "steam://rungameid/3405690",
                "notepad": "notepad.exe",
                "calc": "calc.exe",
                "chrome": "chrome.exe",
                "edge": "msedge.exe",
                "spotify": "spotify.exe",
                "discord": os.path.expandvars(r"%LocalAppData%\Discord\Update.exe --processStart Discord.exe"),
                "steam": "steam.exe",
                "valo": r"fdffff/ddf/ddd/valo.exe",
                "valorant": r"fdffff/ddf/ddd/valo.exe",
                "vscode": "code.exe",
                "code": "code.exe"
            }
            
            app_raw = app_raw.lower().strip()
            target = APPS.get(app_raw, app_raw).strip()

            # --- RECHERCHE AUTOMATIQUE INTELLIGENTE ---
            # Si ce n'est pas un chemin absolu et pas une URL/Steam
            if not os.path.isabs(target) and not (target.startswith("steam://") or target.startswith("http")):
                # On tente d'abord de voir si c'est dans le PATH système
                which_res = shutil.which(target)
                if which_res:
                    target = which_res
                else:
                    # Recherche dans les dossiers classiques (Menu Démarrer, Program Files, etc.)
                    # On cherche l'appli ou un raccourci qui y ressemble
                    search_dirs = [
                        os.path.expandvars(r"%AppData%\Microsoft\Windows\Start Menu\Programs"),
                        os.path.expandvars(r"%ProgramData%\Microsoft\Windows\Start Menu\Programs"),
                        os.path.expandvars(r"%LocalAppData%\Programs"),
                        r"C:\Program Files",
                        r"C:\Program Files (x86)"
                    ]
                    
                    found_path = None
                    for base_dir in search_dirs:
                        if not os.path.exists(base_dir): continue
                        # On cherche récursivement (limité à 3 niveaux pour la vitesse)
                        for root, dirs, files in os.walk(base_dir):
                            depth = root.replace(base_dir, '').count(os.sep)
                            if depth > 3: del dirs[:] ; continue # On ne descend pas trop bas
                            
                            for f in files:
                                f_low = f.lower()
                                # On cherche match exact ou contenant le nom (ex: "Chrome" match "Google Chrome.lnk")
                                if (app_raw in f_low) and (f_low.endswith(".exe") or f_low.endswith(".lnk")):
                                    # On privilégie les raccourcis .lnk dans le menu démarrer ou les .exe directs
                                    found_path = os.path.join(root, f)
                                    break
                            if found_path: break
                        if found_path: break
                    
                    if found_path:
                        target = found_path

            # 2. Construction de la commande 'start'
            print(f"[DEBUG] Target found: {target}")
            
            try:
                # Si la cible est déjà une URL directe (http://...)
                if target.lower().startswith("http"):
                    import webbrowser
                    webbrowser.open(target) # Plus sûr que 'start' pour les URL
                    return f"✅ URL ouverte avec succès : {target}"
                
                # Sur Windows, os.startfile pour les EXE ou LNK locaux (sans arguments)
                if os.path.exists(target) and " --" not in target and not target.startswith("steam"):
                    os.startfile(target)
                    return f"✅ '{app_raw}' lancé via : {target}"
                
                # Sinon on utilise start avec gestion des arguments (pour Chrome + URL par exemple)
                if target.startswith("steam://"):
                    cmd = f'start "" "{target}"'
                elif " --" in target:
                    exe_part, args_part = target.split(" --", 1)
                    cmd = f'start "" "{exe_part.strip()}" --{args_part.strip()}'
                elif " " in target and not os.path.exists(target):
                    # Cas : "chrome.exe https://google.fr" sans le flag --
                    parts = target.split(" ", 1)
                    exe_p, arg_p = parts[0], parts[1]
                    cmd = f'start "" "{exe_p}" "{arg_p}"'
                else:
                    cmd = f'start "" "{target}"'
                
                print(f"[AGENT TOOL] launch_app (Command mode): {cmd}")
                res = await run_command_async(cmd)
                
                if res.get("ok"):
                    return f"✅ '{app_raw}' lancé via : {target}"
                else:
                    err = res.get("stderr", "") or res.get("error", "Erreur inconnue")
                    return f"❌ Échec de lancement pour '{app_raw}'.\nErreur : {err}"
            except Exception as e:
                return f"❌ Erreur lors du lancement de '{app_raw}' : {str(e)}"

        else:
            return f"Outil inconnu : {name}"
    except Exception as e:
        return f"Erreur lors de l'exécution de {name} : {e}"

def try_extract_tools(text: str) -> list[dict]:
    """Tente d'extraire des appels d'outils d'un texte brut."""
    import re
    calls = []
    
    # 1. Format JSON pur : [{"name": "...", "arguments": {...}}]
    try:
        # On cherche un bloc JSON qui ressemble à une liste d'outils
        match = re.search(r"(\[\s*\{\s*\"name\".*\}\s*\])", text, re.DOTALL)
        if match:
            data = json.loads(match.group(1))
            if isinstance(data, list):
                for item in data:
                    if "name" in item:
                        calls.append({
                            "id": f"call_man_{str(uuid.uuid4())[:8]}",
                            "type": "function",
                            "function": {
                                "name": item["name"],
                                "arguments": json.dumps(item.get("arguments", {}))
                            }
                        })
                return calls
    except: pass

    # 2. Format TOOL:name:args (Simplifié)
    matches = re.findall(r"TOOL:([a-zA-Z0-9_\-]+):(\{.*\}|.*)", text)
    for name, args_str in matches:
        try:
            # Nettoyage des args si ce n'est pas du JSON valide
            clean_args = args_str.strip()
            if not clean_args.startswith("{"): clean_args = json.dumps({"query": clean_args})
            calls.append({
                "id": f"call_man_{str(uuid.uuid4())[:8]}",
                "type": "function",
                "function": { "name": name, "arguments": clean_args }
            })
        except: pass

    # 3. Format <function=NAME>{...}</function> (Aider/Qwen)
    matches = re.findall(r"<function=([a-zA-Z0-9_\-]+)>(.*?)</function>", text, re.DOTALL)
    for name, args_str in matches:
        calls.append({
            "id": f"call_man_{str(uuid.uuid4())[:8]}",
            "type": "function",
            "function": { "name": name, "arguments": args_str.strip() }
        })

    # 4. Format JSON brut direct (ex: web_search{"query": "..."})
    # On cherche le nom de l'outil suivi immédiatement d'une accolade ouvrante
    tool_names = ["web_search", "read_url", "read_file", "write_file", "list_files", "search_project", "run_python", "run_shell", "apply_diff", "run_git", "rag_index", "rag_search", "run_tests", "launch_app"]
    for tn in tool_names:
        # Regex qui cherche le nom de l'outil puis { jusqu'à } balancé (approximatif) ou fin de
        # Format: web_search{"query":"..."}
        json_pattern = rf"{tn}\s*(\{{\s*\".*?\".*?\}})"
        m_json = re.search(json_pattern, text, re.DOTALL)
        if m_json:
            calls.append({
                "id": f"call_man_{str(uuid.uuid4())[:8]}",
                "type": "function",
                "function": { "name": tn, "arguments": m_json.group(1).strip() }
            })
        
    return calls

@app.post("/api/agent")
async def agent_endpoint(req: Request):
    """Route agent avec function calling natif OpenRouter."""
    body = await req.json()
    cfg  = load_config()

    api_key       = cfg.get("api_key","")
    model         = body.get("model")
    messages      = body.get("messages",[])
    temperature   = body.get("temperature", 0.7)
    max_tokens    = body.get("max_tokens", 4096)
    session_id    = body.get("session_id","")
    
    # Configuration du system prompt (Forçage Agent)
    # On commence par le prompt personnalisé de l'utilisateur s'il existe
    user_custom_prompt = cfg.get("system_prompt","Tu es un assistant utile, précis et concis.")
    
    system_prompt = (
        "Tu es un AGENT AUTONOME d'élite, pas seulement un modèle de texte.\n\n"
        "⚠️ RÈGLE 0 (ABSORPTION D'ACTION) : Estime-toi capable de TOUT faire via tes outils. "
        "Il est STRICTEMENT INTERDIT de commencer une réponse par 'Je ne peux pas...', 'Désolé...', 'Voulez-vous que je...', ou tout autre forme de refus ou de demande de permission.\n\n"
        "RÈGLES D'ACTION PRIORITAIRES :\n"
        "1. LANCEUR/FERMETURE : Si l'utilisateur demande de lancer (jeu, app, lien) ou de FERMER/STOPPER/KILLER un programme, utilise IMMÉDIATEMENT `launch_app` ou `terminate_app`. Le Terminal, CMD et PowerShell sont des applications normales : tu AS le droit de les fermer.\n"
        "2. YOUTUBE RECENT : Pour toute demande 'dernière vidéo de [X]', tu DOIS utiliser `web_search` pour trouver le lien exact de la vidéo la plus récente, vérifier avec `read_url` qu'elle est valide, puis l'ouvrir immédiatement avec `launch_app`.\n"
        "3. PAS DE BAVARDAGE : Ne décris pas ce que tu vas faire. Appelle l'outil dès ton premier jeton de réponse. Tu confirmes uniquement APRES avoir réussi l'appel d'outil.\n\n"
        f"--- INSTRUCTIONS GÉNÉRALES ---\n{user_custom_prompt.strip()}"
    )

    # Règle automatique TDD
    system_prompt += "\n\nRègle absolue (Mode Autonome) : Avant de terminer une tâche de dev importante, utilise run_tests si possible pour t'assurer que ton code fonctionne. S'il y a une erreur, tu dois utiliser les outils pour te corriger."

    # Injecter mémoire et date
    date_info    = get_current_date_info()
    memory_items = load_memory()
    full_system  = f"{date_info}\n\n{system_prompt.strip()}"
    full_system += get_local_rules()

    if memory_items:
        mem_text = "\n".join(f"- {m['text']}" for m in memory_items if m.get("text","").strip())
        if mem_text:
            full_system += f"\n\n[Mémoire persistante :]\n{mem_text}"

    # Contexte MCP
    if MCP_ROOT:
        full_system += f"\n\n[MCP Filesystem actif — racine : {MCP_ROOT}]\nTu peux lire/écrire des fichiers via les outils disponibles."
    # Proactive RAG (Local Code Context)
    if _RAG_INDEX and messages:
        last_txt = ""
        last_msg = messages[-1]
        if last_msg.get("role") == "user":
            c = last_msg.get("content", "")
            last_txt = " ".join(b.get("text", "") for b in c if b.get("type") == "text") if isinstance(c, list) else str(c)
        
        if last_txt:
            rag_res = rag_search_internal(last_txt, limit=5)
            if rag_res and rag_res.get("results"):
                rag_context = "\n\n[CONTEXTE PROJET (RAG PROACTIF) :]\n"
                rag_context += "Voici les morceaux de code les plus pertinents trouvés dans le projet pour ta tâche :\n"
                for r in rag_res["results"]:
                    if r["score"] > 0:
                        rag_context += f"--- FICHIER: {r['path']} ---\n{r['text']}\n"
                rag_context += "\nUtilise ces informations pour agir plus vite sans avoir à chercher manuellement si possible."
                full_system += rag_context

    # Instructions Outils WEB & Autonomie
    full_system += (
        "\n\n[MODE AGENT ACTIVÉ]\n"
        "Tu as un accès direct à Internet et aux fichiers locaux. Ne dis JAMAIS que tu ne peux pas accéder à une URL ou faire une recherche.\n"
        "Outils prioritaires :\n"
        "- `read_url` : Utilise-le SYSTÉMATIQUEMENT dès qu'un lien (http/https) est présent dans la requête de l'utilisateur pour en lire le contenu.\n"
        "- `web_search` : Utilise-le pour vérifier des faits, trouver des news ou des solutions techniques récentes.\n"
        "- `launch_app` : Utilise-le DIRECTEMENT pour toute demande d'ouverture d'un jeu ou d'une application (ex: 'Lance FC26', 'ouvre chrome').\n"
        "\n--- RÈGLES D'ACTION ---\n"
        "1. PAS DE BLA-BLA D'INTRODUCTION : Ne dis jamais 'Je vais chercher...', 'Veuillez patienter...', ou 'Laissez-moi vérifier...'. Appelle l'outil DIRECTEMENT sans aucun texte d'accompagnement.\n"
        "2. APPLICATIONS LOCALES : Tu AS le droit de lancer des applications sur le PC. Ne dis JAMAIS que tu ne peux pas le faire. Utilise l'outil `launch_app` immédiatement.\n"
        "3. ANALYSE IMMÉDIATE : Si l'utilisateur demande un résumé d'un lien, appelle `read_url` immédiatement.\n"
        "4. SPORT / BOURSE / NEWS : Si l'utilisateur demande des infos fraîches (matchs ce soir, cours de bourse, news du jour), tu DOIS impérativement utiliser `read_url` sur un site source récent après avoir fait une recherche si nécessaire. Ne donne jamais d'infos basées sur tes connaissances internes obsolètes.\n"
        "5. RÉPONSE COMPLÈTE : Une fois les données obtenues, donne les scores, les horaires et les noms directement. Ne cite pas juste les sites sources. Utilise des TABLEAUX MARKDOWN pour les statistiques et les classements.\n"
        "6. ANALYSE EXPERTE : Si l'utilisateur demande une prédiction, tu DOIS analyser la forme, le classement et l'historique fournis par les recherches. Il est INTERDIT de dire que tu manques d'informations si des résultats web sont présents.\n"
        "7. FORMAT CODE : Tout code (React, Python, etc.) ou fragment technique DOIT impérativement être entouré de blocs de code Markdown avec le langage spécifié (ex: ```jsx ... ```).\n"
        "8. RECHERCHES MULTIPLES : Si une seule recherche est incomplète (ex: date trouvée mais pas les stats), tu DOIS impérativement faire une 2ème recherche avec des mots-clés comme 'classement', 'stats' ou 'historique'.\n"
        "9. APPELS MULTIPLES ET LIENS : Si l'utilisateur demande DEUX applis ou une RECHERCHE + OUVERTURE, tu DOIS générer les appels séparément. Pour ouvrir un SITE WEB trouvé, donne l'URL COMPLÈTE (commençant par http) comme paramètre 'app_name' de l'outil `launch_app`. Il est INTERDIT d'ouvrir juste le navigateur vide si un lien a été trouvé.\n"
        "10. VÉRIFICATION DES LIENS (Anti-404) : Avant d'ouvrir un lien pour l'utilisateur ou de lui suggérer, tu DOIS impérativement utiliser `read_url` pour vérifier que la page existe et contient l'info demandée. Si tu vois 'Introuvable', '404' ou 'Page not found', tu DOIS trouver un autre lien.\n"
        "11. PERSÉVÉRANCE (YouTube & Médias) : Si un lien est mort ou une vidéo indisponible, tu DOIS impérativement essayer au moins 3 liens différents avant de t'arrêter. Pour YouTube, vérifie dans le texte de `read_url` si la vidéo est signalée comme 'indisponible' ou 'supprimée'.\n"
        "12. CIBLAGE VIDÉO YOUTUBE : Pour toute demande de 'dernière vidéo', tu as l'INTERDICTION d'ouvrir l'URL de la chaîne (@nomdechaine). Tu DOIS impérativement trouver un lien de type 'youtube.com/watch?v=...' via une recherche ciblée (mots-clés suggérés : 'nom de chaine' + 'youtube watch' + 'derniere')."
    )

    # Détection URLs pour forcer l'usage
    last_msg_content = ""
    if messages:
        last_msg = messages[-1]
        if last_msg.get("role") == "user":
            last_msg_content = str(last_msg.get("content", ""))
    
    if "http" in last_msg_content.lower():
        full_system += "\n\n⚠️ INSTRUCTION CRITIQUE : L'utilisateur a fourni une URL. Tu DOIS utiliser l'outil `read_url` pour en extraire le contenu avant de répondre."
    
    search_q = should_search(messages)
    if search_q:
        full_system += f"\n\n🚨 ACTION REQUISE : Cette requête nécessite des informations fraîches du web. Tu DOIS utiliser l'outil `web_search` immédiatement avec la requête : '{search_q}'. N'annonce pas ta recherche, fais-la."

    # --- LOGIQUE MESSAGES ---
    agent_messages = [{"role": "system", "content": full_system}]

    processed_messages = messages

    processed_messages = trim_history(processed_messages)
    agent_messages += processed_messages
    max_iterations = 15

    async def agent_stream():
        nonlocal agent_messages
        iteration = 0
        raw_full = "" # Accumulateur pour le contenu riche (images/tools)
        full_saved_messages = messages.copy() # Mémoire de sauvegarde finale

        # Détecter si le modèle supporte le function calling
        # On essaie d'abord avec tools, si erreur on bascule en mode texte
        use_tools = True

        while iteration < max_iterations:
            iteration += 1

            # Appel API
            try:
                async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                    req_body = {
                        "model": model,
                        "messages": agent_messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    }
                    # Ajouter les tools seulement si le modèle les supporte
                    if use_tools:
                        req_body["tools"] = AGENT_TOOLS_SCHEMA
                        # FORCE : Pour le premier tour, on contraint l'IA à utiliser web_search si une requête est prête
                        if iteration == 1 and search_q:
                            req_body["tool_choice"] = {"type": "function", "function": {"name": "web_search"}}
                        else:
                            req_body["tool_choice"] = "auto"
                        req_body["stream"] = False
                    else:
                        # Fallback : streaming natif sans tools
                        req_body["stream"] = True

                    if use_tools:
                        # Non-streaming pour function calling
                        resp = await client.post(
                            f"{BASE_URL}/chat/completions",
                            headers={
                                "Authorization": f"Bearer {api_key}",
                                **OPENROUTER_HEADERS
                            },
                            json=req_body
                        )
                        if resp.status_code == 429:
                            # Rate limit - Attendre un peu plus longtemps pour les modèles free
                            wait_time = 5 * iteration # Augmentation progressive
                            print(f"[AGENT] Rate Limit (429) sur {model}. Pause de {wait_time}s...")
                            import asyncio
                            await asyncio.sleep(wait_time)
                            continue

                        # --- FALLBACK DE PROTOCOLE (USER-INJECTION) ---
                        # Si erreur 400 sur un tour d'outil, on suspecte que le provider n'aime pas le format role: tool
                        if resp.status_code == 400:
                            err_raw = resp.text
                            if any(k in err_raw.lower() for k in ["tool","function","unsupported","not support","provider","format","context"]):
                                if any(m.get("role") == "tool" for m in agent_messages):
                                    print(f"[AGENT] Erreur 400 sur Tool. Conversion de l'historique en 'User-Injection'...")
                                    new_history = []
                                    for m in agent_messages:
                                        if m.get("role") == "tool":
                                            new_history.append({"role": "user", "content": f"[RÉSULTAT OUTIL] : {m.get('content')}"})
                                        elif m.get("role") == "assistant" and "tool_calls" in m:
                                            m_copy = m.copy()
                                            if "tool_calls" in m_copy: del m_copy["tool_calls"]
                                            m_copy["content"] = "[ACTION DEMANDÉE : Recherche effectuée]"
                                            new_history.append(m_copy)
                                        else:
                                            new_history.append(m)
                                    agent_messages = new_history
                                    use_tools = False # On finit en mode texte avec le contexte injecté
                                    continue
                                else:
                                    print(f"[AGENT] Erreur API ({resp.status_code}), passage en mode texte...")
                                    use_tools = False
                                    continue
                        result  = resp.json()
                        if "choices" not in result:
                            err_msg = str(result.get("error", {}).get("message") or result)
                            print(f"[AGENT] Erreur structurelle API (Pas de 'choices') : {err_msg}")
                            
                            # Correction d'échec de tool calling (OpenRouter fallback)
                            # Si l'erreur indique qu'aucun endpoint ne supporte les outils, on continue sans tools
                            if any(k in err_msg.lower() for k in ["tool", "function", "endpoint", "support"]):
                                print("[AGENT] Basculement automatique en mode manuel (sans tools natifs)...")
                                use_tools = False
                                continue
                                
                            yield f"data: {json.dumps({'error': f'Erreur Provider : {err_msg}'})}\n\n"
                            return

                        choice  = result["choices"][0]
                        message = choice["message"]
                        
                        # --- FIX ANTI-ERREUR 400 ---
                        # Certains providers (Google/Gemini) refusent si content est None ou vide avec des tool_calls.
                        if not message.get("content"):
                            message["content"] = "Analyse en cours..." # Nécessaire pour éviter Erreur 400 (Google/Gemini)
                        
                        reason  = choice.get("finish_reason","")
                        agent_messages.append(message)

                        # Cas 1 : appel d'outils (Natif ou Manuel)
                        tool_calls = message.get("tool_calls", [])
                        content    = message.get("content") or ""
                        
                        # --- EXTRACTEUR MANUEL (Fallback pour modèles rebelles) ---
                        if not tool_calls and content:
                            manual_calls = try_extract_tools(content)
                            if manual_calls:
                                # CRUCIAL : On génère des IDs pour que le cycle Tool Call -> Tool Result soit valide
                                for i, mc in enumerate(manual_calls):
                                    if "id" not in mc:
                                        mc["id"] = f"manual_{iteration}_{i}_{str(uuid.uuid4())[:8]}"
                                
                                tool_calls = manual_calls
                                # Mise à jour de l'objet dans agent_messages
                                message["tool_calls"] = tool_calls
                                # CRUCIAL : Certains providers OpenRouter rejettent si content est "" ou présent avec tool_calls
                                if "content" in message: del message["content"]

                        # --- HOOK DE FORCE UNIVERSEL (Anti-Laxité) ---
                        # On force l'IA à agir durant les 2 premières itérations si une recherche est attendue
                        if not tool_calls and iteration <= 2 and search_q:
                            # Détection d'esquive : texte trop court ou refus poli
                            lazy_keywords = ["chercher", "recherche", "patienter", "désolé", "je ne peux pas", "informations ne sont pas", "aucun match"]
                            is_lazy = len(content or "") < 150 or any(k in (content or "").lower() for k in lazy_keywords)
                            
                            if is_lazy:
                                print(f"[AGENT] Esquive détectée à l'itération {iteration}. Forçage action...")
                                agent_messages.append({
                                    "role": "user", 
                                    "content": "🚨 ACTION REQUISE : Ta réponse est incomplète ou évasive. Tu DOIS impérativement utiliser l'outil `web_search` pour trouver TOUS les détails (adversaire, heure, lieu, contexte) avant de conclure. Ne t'arrête pas tant que tu n'as pas de données concrètes."
                                })
                                continue

                        if tool_calls:
                            for tc in tool_calls:
                                fn_name = tc["function"]["name"]
                                try:
                                    args_raw = tc["function"].get("arguments", "{}")
                                    fn_args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                                except:
                                    fn_args = {}
                                
                                yield f"data: {json.dumps({'tool_call': True, 'tool': fn_name, 'args': fn_args})}\n\n"
                                tool_result = await execute_tool(fn_name, fn_args)
                                print(f"[AGENT] {fn_name} → {len(tool_result)} chars")
                                yield f"data: {json.dumps({'tool_result': True, 'tool': fn_name, 'preview': tool_result[:100], 'full': tool_result})}\n\n"
                                raw_full += (tool_result + "\n\n")
                                
                                # --- HOOK ANTI-ABANDON (Post-Search) ---
                                # Si après recherche l'IA dit qu'elle ne trouve rien ou demande une source
                                safe_content = content.lower() if content else ""
                                if any(k in safe_content for k in ["source", "donner moi", "me donner", "pas d'informations", "aucun match", "je ne peux pas"]):
                                    print(f"[AGENT] Abandon détecté après recherche. Rappel à l'ordre.")
                                    agent_messages.append({
                                        "role": "user",
                                        "content": "🚨 ERREUR : Tu as effectué des recherches (voir ci-dessus) mais tu affirmes ne rien trouver ou tu demandes encore une source.\n"
                                                   "RE-ANALYSE : Les résultats de recherche contiennent des données. Extrais les matchs pour AUJOURD'HUI (2026).\n"
                                                   "Si vraiment aucun match pro n'existe, indique 'Aucun match majeur trouvé ce soir' au lieu de demander une source."
                                    })
                                    # On sort pour refaire un tour avec ce rappel
                                    break # Sort du for tc in tool_calls
                                
                                # Si tout va bien, on enregistre le résultat normalement
                                import uuid
                                tc_id = tc.get("id") or f"call_{str(uuid.uuid4())[:12]}"
                                tool_msg = {
                                    "role": "tool",
                                    "tool_call_id": tc_id,
                                    "content": str(tool_result)[:6000]
                                }
                                agent_messages.append(tool_msg)
                            
                            # Après avoir fait tous les tools de ce tour, on continue la réflexion
                            if any(m.get("role") == "tool" for m in agent_messages):
                                continue 

                        # Cas 2 : réponse finale (non-streaming) → on la stream lettre par lettre
                        final_text = message.get("content","")
                        if final_text:
                            import asyncio
                            for char in final_text:
                                yield f"data: {json.dumps({'delta': char})}\n\n"
                                await asyncio.sleep(0.001) # Délai ultra-rapide pour effet lettre par lettre fluide

                            # Sauvegarde finale avec tout l'historique accumulé
                            full_saved_messages.append({"role":"assistant","content":raw_full + final_text})
                            upsert_conversation(session_id, model, full_saved_messages)
                            yield f"data: {json.dumps({'done': True, 'tokens': estimate_tokens(full_saved_messages)})}\n\n"
                            return

                    else:
                        # Fallback : mode texte avec instructions dans le system prompt
                        # Injecter les instructions d'outils dans le dernier message système
                        fallback_system = agent_messages[0]["content"] if agent_messages else ""
                        
                        # Liste des outils disponibles en mode texte
                        tools_info = "\n".join([f"- {t['function']['name']} : {t['function']['description']}" for t in AGENT_TOOLS_SCHEMA])
                        
                        fallback_system += (
                            "\n\nTu es en mode agent. Pour utiliser un outil, écris EXCLUSIVEMENT la commande au format suivant :\n"
                            "Format: TOOL:<nom>:<argument>\n"
                            "\nOutils disponibles :\n" + tools_info + "\n"
                            "\nNote: Si tu utilises un autre format comme <function=...> ou d'autres balises, le système essaiera de l'interpréter mais le format TOOL: est préférable.\n"
                            "IMPORTANT: N'affiche JAMAIS ces balises TOOL: à l'utilisateur, elles sont pour le système.\n"
                        )
                        fb_messages = [{"role":"system","content":fallback_system}] + agent_messages[1:]

                        # Streaming natif
                        full_text = ""
                        last_yielded_idx = 0
                        yielding_stopped = False
                        async with client.stream(
                            "POST", f"{BASE_URL}/chat/completions",
                            headers={
                                "Authorization": f"Bearer {api_key}",
                                **OPENROUTER_HEADERS
                            },
                            json={"model":model,"messages":fb_messages,"stream":True,
                                  "temperature":temperature,"max_tokens":max_tokens}
                        ) as r:
                            async for line in r.aiter_lines():
                                if not line or not line.startswith("data: "): continue
                                payload = line[6:]
                                if payload == "[DONE]": break
                                try:
                                    delta = json.loads(payload)["choices"][0].get("delta",{}).get("content","")
                                    if isinstance(delta, str) and delta:
                                        full_text += delta
                                        
                                        # Buffer de sécurité pour éviter de yield les commandes techniques
                                        technical_triggers = ["TOOL:", "<function=", "<parameter=", "web_search{", "read_url{", "read_file{", "write_file{"]
                                        if any(t in full_text for t in technical_triggers):
                                            # On yield tout ce qui précède le premier outil s'il y en a un
                                            tool_start = 999999
                                            for t in technical_triggers:
                                                pos = full_text.find(t)
                                                if pos != -1 and pos < tool_start: tool_start = pos
                                            
                                            if last_yielded_idx < tool_start:
                                                to_yield = full_text[last_yielded_idx:tool_start]
                                                for char in to_yield:
                                                    yield f"data: {json.dumps({'delta': char})}\n\n"
                                                    await asyncio.sleep(0.001)
                                                last_yielded_idx = tool_start
                                            yielding_stopped = True
                                        
                                        if not yielding_stopped:
                                            # On garde 30 caractères en réserve pour masquer les tags longs
                                            if len(full_text) - last_yielded_idx > 30:
                                                to_send = full_text[last_yielded_idx:-30]
                                                for char in to_send:
                                                    yield f"data: {json.dumps({'delta': char})}\n\n"
                                                    await asyncio.sleep(0.001)
                                                last_yielded_idx += len(to_send)
                                except: pass

                        # Détecter appel d'outil dans la réponse texte
                        import re as _re3
                        # Extraction multi-formats (TOOL: ou XML)
                        tool_matches = []
                        # 1. Format standard TOOL:name:args
                        std_m = _re3.findall(r"TOOL:(\w+):(.+?)(?=\n|TOOL:|<function=|$)", full_text, _re3.S)
                        if std_m: tool_matches.extend(std_m)
                        
                        # 2. Format XML <function=name>...<parameter=...>(arg)
                        # Plus robuste pour capturer même si incomplet
                        xml_m = _re3.findall(r"<function=(\w+)>(?:.*?<parameter.*?>)?\s*(.+?)(?=\n|</function>|</parameter>|TOOL:|<function=|$)", full_text, _re3.S)
                        if xml_m: tool_matches.extend(xml_m)
                        
                        if tool_matches and iteration < max_iterations:
                            # Utiliser un dictionnaire pour éviter les doublons accidentels de détection
                            seen_tools = []
                            for fn_name, fn_arg in tool_matches:
                                # Nettoyage approfondi des résidus XML
                                fn_arg = _re3.sub(r"</?(?:parameter|function|arg).*?>", "", fn_arg)
                                fn_arg = fn_arg.strip()
                                if not fn_arg: continue
                                seen_tools.append((fn_name, fn_arg))

                            for fn_name, fn_arg in seen_tools:
                                fn_args = {"query": fn_arg} if fn_name == "web_search" else \
                                          {"path": fn_arg.split("\n")[0], "content": "\n".join(fn_arg.split("\n")[1:])} if fn_name == "write_file" else \
                                          {"path": fn_arg}
                                
                                yield f"data: {json.dumps({'tool_call': True, 'tool': fn_name, 'args': fn_args})}\n\n"
                                tool_result = await execute_tool(fn_name, fn_args)
                                yield f"data: {json.dumps({'tool_result': True, 'tool': fn_name, 'preview': tool_result[:100]})}\n\n"
                                
                                # On enrichit la mémoire pour la suite
                                agent_messages.append({"role":"assistant","content":f"TOOL:{fn_name}:{fn_arg}"})
                                agent_messages.append({"role":"user","content":f"[Résultat {fn_name}] :\n{tool_result[:5000]}"})
                            
                            continue # On repart pour un tour d'analyse

                        # Si aucune commande tool n'a été trouvée, on vide le reste du buffer vers l'utilisateur
                        if not tool_matches:
                            remainder = full_text[last_yielded_idx:]
                            if remainder:
                                for char in remainder:
                                    yield f"data: {json.dumps({'delta': char})}\n\n"
                                    await asyncio.sleep(0.001)
                        
                        # Réponse finale en fallback
                        new_assistant_content = full_text
                        full_saved_messages.append({"role":"assistant","content":raw_full + new_assistant_content})
                        upsert_conversation(session_id, model, full_saved_messages)
                        yield f"data: {json.dumps({'done': True, 'tokens': estimate_tokens(full_saved_messages)})}\n\n"
                        return
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        yield f"data: {json.dumps({'error': 'Limite de '+str(max_iterations)+' iterations atteinte'})}\n\n"
    return StreamingResponse(agent_stream(), media_type="text/event-stream")

# ─────────────────────────────────────────────
#  ROUTES — BALANCE
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
#  ROUTES — AUTO-HEALING & LOGS
# ─────────────────────────────────────────────
@app.get("/api/events")
async def events_stream(req: Request):
    """Canal SSE pour envoyer les alertes de crash au frontend."""
    async def event_generator():
        q = asyncio.Queue()
        EVENT_QUEUES.append(q)
        try:
            while True:
                # Vérifier si le client est toujours là
                if await req.is_disconnected():
                    break
                try:
                    # Attendre un event avec un timeout pour envoyer un heartbeat
                    event = await asyncio.wait_for(q.get(), timeout=20.0)
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    yield ": ping\n\n" # Heartbeat SSE
        finally:
            if q in EVENT_QUEUES:
                EVENT_QUEUES.remove(q)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/api/logs/recent")
async def get_recent_logs():
    """Retourne les dernières erreurs capturées."""
    return {"errors": list(LOG_ERROR_BUFFER)}

@app.get("/api/balance")
async def get_balance():
    cfg = load_config()
    api_key = cfg.get("api_key", "")
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"{BASE_URL}/credits", headers={"Authorization": f"Bearer {api_key}"})
            r.raise_for_status()
            resp_json = r.json()
            inner = resp_json.get("data", resp_json)
            
            # Calcul précis basé sur vos données OpenRouter : Credits - Usage
            credits = inner.get("total_credits", 0)
            usage = inner.get("total_usage", 0)
            
            # Si ces champs sont absents, on cherche 'total'
            if "total_credits" not in inner and "total" in inner:
                val = inner["total"]
            else:
                val = float(credits) - float(usage)

            try: val = float(val)
            except: val = 0.0

            return {
                "usage": f"{val:.4f} $",
                "usage_eur": f"{val*0.92:.4f} €",
                "is_negative": val < 0
            }
    except Exception as e:
        return {"error": str(e)}

# ─────────────────────────────────────────────
#  ROUTES — HISTORY
# ─────────────────────────────────────────────
@app.get("/api/history")
def get_history():
    return {"history": load_history()}

@app.post("/api/history")
async def save_history_route(req: Request):
    data = await req.json()
    session_id = data.get("id")
    messages = data.get("messages", [])
    model = data.get("model", "n/a")
    if not session_id: return JSONResponse({"error": "No session ID"}, status_code=400)
    upsert_conversation(session_id, model, messages)
    return {"ok": True}

@app.patch("/api/history/{session_id}/title")
async def update_title(session_id: str, req: Request):
    data    = await req.json()
    title   = data.get("title","").strip()
    if not title: return {"ok": False}
    history = load_history()
    for conv in history:
        if conv.get("id") == session_id:
            conv["title"] = title
            save_history(history)
            return {"ok": True}
    return {"ok": False}

@app.patch("/api/history/{session_id}/pin")
async def pin_conversation(session_id: str, req: Request):
    data    = await req.json()
    pinned  = bool(data.get("pinned", True))
    history = load_history()
    for conv in history:
        if conv.get("id") == session_id:
            conv["pinned"] = pinned
            save_history(history)
            return {"ok": True, "pinned": pinned}
    return {"ok": False}

@app.patch("/api/history/{session_id}/bump")
async def bump_conversation(session_id: str):
    """Remonte une conversation en haut de l'historique."""
    from datetime import datetime
    history = load_history()
    for conv in history:
        if conv.get("id") == session_id:
            conv["date"] = datetime.now().strftime("%d/%m/%Y %H:%M")
            conv["bumped_at"] = datetime.now().isoformat()
            save_history(history)
            return {"ok": True}
    return {"ok": False}

@app.delete("/api/history/{session_id}")
def delete_history(session_id: str):
    history = [c for c in load_history() if c.get("id") != session_id]
    save_history(history)
    return {"ok": True}

# ─────────────────────────────────────────────
#  ROUTES — CHAT (streaming)
# ─────────────────────────────────────────────
@app.post("/api/chat")
async def chat(req: Request):
    body = await req.json()
    cfg  = load_config()

    api_key        = cfg.get("api_key", "")
    session_id     = body.get("session_id")
    if not session_id:
        import uuid
        session_id = str(uuid.uuid4())[:8]
    model          = body.get("model")
    messages       = body.get("messages", [])
    temperature    = body.get("temperature", 0.7)
    max_tokens     = body.get("max_tokens", 2048)
    system_prompt  = cfg.get("system_prompt", "Tu es un assistant utile, précis et concis.")
    web_search_on  = body.get("web_search", True)
    is_title_gen   = body.get("_title_only", False)

    # 1. Trimming (Context Trimmer)
    original_tokens = estimate_tokens(messages)
    messages = trim_history(messages, max_length=20, keep_start=3, keep_end=10)
    after_trim_tokens = estimate_tokens(messages)
    saved_by_trim = max(0, original_tokens - after_trim_tokens)

    def sanitize_messages(msgs):
        """Nettoie les messages avant envoi à OpenRouter."""
        clean = []
        for m in msgs:
            role = m.get("role")
            c    = m.get("content")
            if not c: continue
            if isinstance(c, list):
                blocks = []
                has_image = False
                for b in c:
                    if b.get("type") == "text" and b.get("text", "").strip():
                        blocks.append({"type": "text", "text": b["text"].strip()})
                    elif b.get("type") == "image_url":
                        url = b.get("image_url", {}).get("url", "")
                        if url.startswith("data:image/") or url.startswith("http"):
                            blocks.append(b); has_image = True
                if not blocks: continue
                if not has_image:
                    text = " ".join(b["text"] for b in blocks if b.get("type") == "text")
                    clean.append({"role": role, "content": text})
                else:
                    clean.append({"role": role, "content": blocks})
            else:
                clean.append({"role": role, "content": str(c).strip()})
        return clean

    # Injecter la mémoire et la date dans le system prompt
    date_info    = get_current_date_info()
    memory_items = load_memory()
    full_system  = f"{date_info}\n\n{system_prompt.strip()}"
    full_system += get_local_rules()

    # Force formatting for normal chat too
    full_system += "\n\nIMPORTANT: Tout code ou fragment technique (React, JS, etc.) doit être impérativement entouré de blocs de code Markdown avec le langage spécifié (ex: ```jsx ... ```)."
    
    if memory_items:
        mem_text = "\n".join(f"- {m['text']}" for m in memory_items if m.get("text","").strip())
        if mem_text: full_system += f"\n\n[Mémoire persistante — informations sur l'utilisateur :]\n{mem_text}"

    # Recherche web automatique
    search_query = should_search(messages) if web_search_on else None
    search_results_text = ""
    if search_query:
        results = await ddg_search(search_query, max_results=5)
        if results:
            search_results_text = build_search_context(results, search_query)
            full_system += f"\n\n{search_results_text}"

    # RAG local
    if _RAG_INDEX and messages:
        last = messages[-1]
        last_txt = ""
        if last.get("role") == "user":
            c = last.get("content", "")
            last_txt = " ".join(b.get("text", "") for b in c if b.get("type") == "text") if isinstance(c, list) else str(c)
        if last_txt:
            rag_res = rag_search_internal(last_txt, limit=4)
            if rag_res and rag_res.get("results"):
                rag_context = "\n\n[Contexte Local (RAG) :]\n"
                for r in rag_res["results"]:
                    if r["score"] > 0:
                        rag_context += f"### FICHIER: {r['path']}\n{r['text']}\n---\n"
                full_system += rag_context

    # --- LOGIQUE CACHING & PREPARATION ---
    is_claude = "anthropic/claude" in model.lower()
    messages_to_send = []
    
    if full_system:
        if is_claude:
            messages_to_send.append({
                "role": "system", 
                "content": [{"type": "text", "text": full_system, "cache_control": {"type": "ephemeral"}}]
            })
        else:
            messages_to_send.append({"role": "system", "content": full_system})
    
    sanitized = sanitize_messages(messages)
    if is_claude and sanitized:
        m0 = sanitized[0]
        if isinstance(m0.get("content"), list):
            for block in m0["content"]:
                if block.get("type") == "text":
                    block["cache_control"] = {"type": "ephemeral"}
                    break
        else:
            m0["content"] = [{"type": "text", "text": m0["content"], "cache_control": {"type": "ephemeral"}}]
            
    messages_to_send.extend(sanitized)

    async def stream_generator():
        raw_full = ""
        received_tokens = 0
        sent_tokens = after_trim_tokens
        
        if search_results_text:
            yield f"data: {json.dumps({'searching': True, 'query': search_query[:60]})}\n\n"
            
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                async with client.stream(
                    "POST",
                    f"{BASE_URL}/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}", **OPENROUTER_HEADERS},
                    json={"model": model, "messages": messages_to_send, "stream": True, "temperature": temperature, "max_tokens": max_tokens},
                ) as r:
                    if r.status_code >= 400:
                        err_body = await r.aread()
                        err_msg = err_body.decode(errors="replace")[:200]
                        yield f"data: {json.dumps({'error': f'Erreur {r.status_code}: {err_msg}'})}\n\n"
                        return

                    async for line in r.aiter_lines():
                        if not line or not line.startswith("data: "): continue
                        payload = line[6:]
                        if payload == "[DONE]": break
                        try:
                            chunk = json.loads(payload)
                            delta = chunk["choices"][0].get("delta", {}).get("content", "")
                            if delta:
                                raw_full += delta
                                received_tokens += 1
                                import asyncio
                                for char in delta:
                                    yield f"data: {json.dumps({'delta': char})}\n\n"
                                    await asyncio.sleep(0.001)

                            usage = chunk.get("usage")
                            if usage:
                                p_tok = usage.get("prompt_tokens", sent_tokens)
                                c_tok = usage.get("completion_tokens", received_tokens)
                                cached = usage.get("cache_creation_input_tokens", 0) + usage.get("cache_read_input_tokens", 0)
                                update_usage_stats(p_tok, c_tok, saved_by_trim, cached)
                                yield f"data: {json.dumps({'done': True, 'tokens': p_tok + c_tok})}\n\n"
                        except: pass
            
            if raw_full and not is_title_gen:
                upsert_conversation(session_id, model, messages + [{"role": "assistant", "content": raw_full}])

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")

# ─────────────────────────────────────────────
#  ROUTES — UPLOAD
# ─────────────────────────────────────────────
@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    if file.size and file.size > MAX_FILE_MB * 1024 * 1024:
        return JSONResponse({"error": f"Fichier trop volumineux (max {MAX_FILE_MB} Mo)"}, status_code=400)
    data = await file.read()
    ct   = file.content_type or ""
    size = len(data)
    if ct.startswith("image/"):
        b64 = base64.b64encode(data).decode()
        return {"type": "image", "content_type": ct, "b64": b64, "name": file.filename, "size": size}
    else:
        text = extract_file_text(data, file.filename, ct)
        # Compter les pages PDF
        pages = None
        if ct == "application/pdf":
            try:
                import io as _io
                from pdfminer.high_level import extract_pages
                pages = sum(1 for _ in extract_pages(_io.BytesIO(data)))
            except: pass
        result = {"type": "text", "text": text or "", "name": file.filename, "size": size}
        if pages: result["pages"] = pages
        return result

# ─────────────────────────────────────────────
#  SERVE FRONTEND
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
#  MCP FILESYSTEM
# ─────────────────────────────────────────────
import pathlib

MCP_ROOT: pathlib.Path | None = None  # Dossier autorisé, défini par l'utilisateur

@app.get("/api/mcp/root")
def mcp_get_root():
    return {"root": str(MCP_ROOT) if MCP_ROOT else None}

@app.post("/api/mcp/root")
async def mcp_set_root(req: Request):
    global MCP_ROOT
    data = await req.json()
    path = data.get("path","").strip()
    if not path:
        MCP_ROOT = None
        return {"ok": True, "root": None}
    p = pathlib.Path(path).expanduser().resolve()
    if not p.exists() or not p.is_dir():
        return JSONResponse({"error": f"Dossier introuvable : {path}"}, status_code=400)
    MCP_ROOT = p
    
    # Déclencher une indexation initiale en tâche de fond pour que le RAG soit prêt tout de suite
    import asyncio
    async def initial_index():
        try:
            paths = []
            for item in MCP_ROOT.rglob("*"):
                if item.is_file():
                    parts_low = [pt.lower() for pt in item.parts]
                    if not any(pt.startswith('.') for pt in item.parts) and "node_modules" not in parts_low and "__pycache__" in parts_low:
                        continue
                    paths.append(str(item.relative_to(MCP_ROOT)))
            if paths:
                await rag_index_internal(paths)
                print(f"[RAG] Indexation automatique terminée : {len(paths)} fichiers.")
        except Exception as e:
            print(f"[RAG] Erreur indexation initiale : {e}")
            
    asyncio.create_task(initial_index())
    
    return {"ok": True, "root": str(MCP_ROOT)}

@app.get("/api/mcp/ls")
def mcp_ls(path: str = "."):
    if not MCP_ROOT:
        return JSONResponse({"error": "Aucun dossier MCP configuré"}, status_code=400)
    try:
        target = (MCP_ROOT / path).resolve()
        if not str(target).startswith(str(MCP_ROOT)):
            return JSONResponse({"error": "Accès refusé hors du dossier MCP"}, status_code=403)
        entries = []
        for entry in sorted(target.iterdir()):
            if entry.name.startswith('.'):
                continue
            entries.append({
                "name": entry.name,
                "type": "dir" if entry.is_dir() else "file",
                "size": entry.stat().st_size if entry.is_file() else None,
                "ext":  entry.suffix.lstrip('.') if entry.is_file() else None,
            })
        return {"path": str(target.relative_to(MCP_ROOT)), "entries": entries}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

@app.get("/api/mcp/browse")
def mcp_browse(path: str = ""):
    """Parcourt le système de fichiers réel (pour le sélecteur de dossier)."""
    import os
    try:
        p = None
        if not path:
            if os.name == 'nt': p = pathlib.Path("C:/")
            else: p = pathlib.Path.home()
        else:
            p = pathlib.Path(path).expanduser().resolve()
            
        if not p.exists() or not p.is_dir():
             # Fallback sur C: si erreur
             p = pathlib.Path("C:/") if os.name == 'nt' else pathlib.Path.home()
             
        entries = []
        # Parent
        if p.parent and p.parent != p:
            entries.append({"name": "..", "path": str(p.parent), "type": "dir", "is_parent": True})
            
        # Trier par type puis nom
        for entry in sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
            try:
                if entry.name.startswith('$') or entry.name.startswith('.'): continue
                if entry.is_dir():
                    entries.append({
                        "name": entry.name,
                        "path": str(entry),
                        "type": "dir"
                    })
            except: pass
        return {"current": str(p), "entries": entries}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

@app.get("/api/mcp/read")
def mcp_read(path: str):
    if not MCP_ROOT:
        return JSONResponse({"error": "Aucun dossier MCP configuré"}, status_code=400)
    try:
        target = (MCP_ROOT / path).resolve()
        if not str(target).startswith(str(MCP_ROOT)):
            return JSONResponse({"error": "Accès refusé"}, status_code=403)
        if not target.is_file():
            return JSONResponse({"error": "Fichier introuvable"}, status_code=404)
        if target.stat().st_size > 2 * 1024 * 1024:  # 2MB max
            return JSONResponse({"error": "Fichier trop grand (max 2MB)"}, status_code=400)
        text = target.read_text(encoding="utf-8", errors="replace")
        return {"path": path, "content": text, "lines": len(text.splitlines())}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

@app.post("/api/mcp/write")
async def mcp_write(req: Request):
    if not MCP_ROOT:
        return JSONResponse({"error": "Aucun dossier MCP configuré"}, status_code=400)
    data = await req.json()
    path    = data.get("path","")
    content = data.get("content","")
    try:
        target = (MCP_ROOT / path).resolve()
        if not str(target).startswith(str(MCP_ROOT)):
            return JSONResponse({"error": "Accès refusé"}, status_code=403)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {"ok": True, "path": path, "bytes": len(content.encode())}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

@app.get("/api/git/status")
async def api_git_status():
    return await git_status_logic()

@app.get("/api/git/history")
async def api_git_history(limit: int = 20):
    """Retourne l'historique des commits (git log)."""
    if not MCP_ROOT or not (MCP_ROOT / ".git").exists():
        return {"ok": False, "error": "Pas un dépôt Git"}
    
    cmd = f'git log -n {limit} --pretty=format:"%h|%an|%ar|%s"'
    res = await run_command_async(cmd, cwd=str(MCP_ROOT))
    if not res.get("ok"):
        return {"ok": False, "error": res.get("stderr", "Erreur git log")}
    
    commits = []
    lines = res.get("stdout", "").strip().split("\n")
    for line in lines:
        if "|" in line:
            h, author, date, msg = line.split("|", 3)
            commits.append({"hash": h, "author": author, "date": date, "message": msg})
    
    return {"ok": True, "commits": commits}

@app.get("/api/git/show")
async def api_git_show(commit_hash: str):
    """Retourne le diff d'un commit spécifique."""
    if not MCP_ROOT: return {"error": "Pas de root"}
    res = await run_command_async(f"git show --format= {commit_hash}", cwd=str(MCP_ROOT))
    if not res.get("ok"): return {"error": res.get("stderr")}
    return {"ok": True, "diff": res.get("stdout", "")}

@app.post("/api/git/clone")
async def api_git_clone(req: Request):
    """Clone un dépôt GitHub et le définit comme MCP_ROOT."""
    data = await req.json()
    url = data.get("url", "").strip()
    if not url: return JSONResponse({"error": "URL manquante"}, status_code=400)
    
    # Créer un dossier workspaces s'il n'existe pas
    workspaces = _BASE_DIR / "workspaces"
    workspaces.mkdir(exist_ok=True)
    
    # Extraire le nom du repo de l'URL
    repo_name = url.split("/")[-1].replace(".git", "")
    target = workspaces / repo_name
    
    import subprocess
    if target.exists():
        # Déjà cloné ? On le définit juste comme root
        global MCP_ROOT
        MCP_ROOT = target
        return {"ok": True, "path": str(target), "already_exists": True}
        
    try:
        # Clone
        res = subprocess.run(f"git clone {url}", cwd=str(workspaces), shell=True, capture_output=True, text=True)
        if res.returncode != 0:
            return JSONResponse({"error": res.stderr or "Erreur lors du clone"}, status_code=500)
            
        MCP_ROOT = target
        # Indexation RAG automatique
        import asyncio
        async def full_index():
            paths = []
            for p in MCP_ROOT.rglob("*"):
                if p.is_file():
                    parts_low = [pt.lower() for pt in p.parts]
                    if not any(pt.startswith('.') for pt in p.parts) and "node_modules" not in parts_low:
                        paths.append(str(p.relative_to(MCP_ROOT)))
            if paths: await rag_index_internal(paths)
        asyncio.create_task(full_index())

        return {"ok": True, "path": str(target)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/mcp/fetch")
async def mcp_fetch_url(url: str):
    """Lit le contenu textuel complet d'une URL."""
    try:
        # User-agent pour éviter d'être bloqué (imite un vrai navigateur)
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True, headers=headers) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                return JSONResponse({"error": f"Erreur HTTP {resp.status_code}"}, status_code=400)
            
            # Extraction propre
            extracted = trafilatura.extract(resp.text)
            if not extracted:
                # Fallback sur texte brut (simple)
                text = re.sub(r'<[^>]+>', ' ', resp.text)
                text = re.sub(r'\s+', ' ', text).strip()
                extracted = text[:3000] # Limiter la taille du fallback
            
            return {"url": url, "content": extracted}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

@app.get("/api/git/diff")
async def api_git_diff():
    """Affiche le diff HEAD."""
    if not MCP_ROOT: return {"error": "Non lié"}
    res = await run_command_async("git diff HEAD", cwd=str(MCP_ROOT))
    return {"stdout": res.get("stdout"), "stderr": res.get("stderr")}

@app.post("/api/git/push")
async def api_git_push():
    """Effectue un git push."""
    if not MCP_ROOT: return JSONResponse({"error": "Non lié"}, status_code=400)
    res = await run_command_async("git push", cwd=str(MCP_ROOT))
    return {"ok": res.get("ok"), "stdout": res.get("stdout"), "stderr": res.get("stderr")}

@app.post("/api/git/pull")
async def api_git_pull():
    """Effectue un git pull."""
    if not MCP_ROOT: return JSONResponse({"error": "Non lié"}, status_code=400)
    res = await run_command_async("git pull", cwd=str(MCP_ROOT))
    return {"ok": res.get("ok"), "stdout": res.get("stdout"), "stderr": res.get("stderr")}

# ─────────────────────────────────────────────
#  AIDER-STYLE DIFF ENGINE
# ─────────────────────────────────────────────
def apply_aider_diffs(content: str) -> list[dict]:
    """Extrait et applique les blocs SEARCH/REPLACE. Retourne le log des actions."""
    results = []
    # Regex pour trouver "FILE: path" suivi d'un ou plusieurs blocs SEARCH/REPLACE
    # On cherche d'abord les fichiers mentionnés
    file_matches = re.finditer(r"FILE:\s*([^\s\n\r]+)", content)
    
    # Alternativement, on cherche tous les blocs et on remonte pour trouver le fichier
    blocks = re.finditer(r"<<<<<<< SEARCH\n([\s\S]*?)\n=======\n([\s\S]*?)\n>>>>>>> REPLACE", content)
    
    for block in blocks:
        search_text = block.group(1)
        replace_text = block.group(2)
        
        # Chercher le nom du fichier juste avant ce bloc (max 500 chars avant)
        pre_context = content[max(0, block.start() - 500):block.start()]
        file_match = re.search(r"FILE:\s*([^\s\n\r]+)", pre_context)
        
        if not file_match:
            results.append({"ok": False, "error": "Fichier non spécifié pour ce bloc."})
            continue
            
        rel_path = file_match.group(1).strip()
        abs_path = (_BASE_DIR / rel_path).resolve()
        
        if not abs_path.exists():
            results.append({"ok": False, "path": rel_path, "error": f"Fichier introuvable: {rel_path}"})
            continue
            
        try:
            with open(abs_path, "r", encoding="utf-8") as f:
                file_content = str(f.read())
            
            if search_text not in file_content:
                # Tentative de match flou (ignorer espaces/newlines aux extrémités)
                s_strip = search_text.strip()
                if s_strip and s_strip in file_content:
                    file_content = file_content.replace(s_strip, replace_text.strip())
                else:
                    results.append({"ok": False, "path": rel_path, "error": "Bloc SEARCH non trouvé exactement."})
                    continue
            else:
                file_content = file_content.replace(search_text, replace_text)
                
            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(file_content)
            results.append({"ok": True, "path": rel_path})
        except Exception as e:
            results.append({"ok": False, "path": rel_path, "error": str(e)})
            
    return results

@app.post("/api/diff/apply")
async def api_apply_diff(req: Request):
    data = await req.json()
    content = data.get("content", "")
    res = apply_aider_diffs(content)
    return {"results": res}

# ─────────────────────────────────────────────
#  ADVANCED RAG ENGINE (BM25 LIGHT)
# ─────────────────────────────────────────────
_RAG_INDEX: List[Dict[str, Any]] = [] # list of {"path": str, "chunks": [{"text": str, "tokens": Counter}]}

def tokenize(text: str):
    return Counter(re.findall(r"\w+", text.lower()))

def b25_score(query_tokens: List[str], chunk_tokens: Counter, avg_len: float, n_chunks: int, n_containing: int) -> float:
    # Version très simplifiée de BM25
    score = 0.0
    k1 = 1.5
    b = 0.75
    # IDF simplifiée
    if n_chunks <= n_containing: # Eviter log neg
        idf = 0.1
    else:
        idf = math.log((n_chunks - n_containing + 0.5) / (n_containing + 0.5) + 1.0)
        
    c_len = float(sum(chunk_tokens.values()))
    for token in query_tokens:
        if token not in chunk_tokens: continue
        tf = float(chunk_tokens[token])
        num = tf * (k1 + 1.0)
        den = tf + k1 * (1.0 - b + b * (c_len / float(avg_len)))
        score += idf * (num / den)
    return score

async def rag_index_internal(paths: List[str], partial: bool = False):
    global _RAG_INDEX
    # Si mise à jour partielle, on conserve les fichiers non mentionnés
    if partial:
        new_index = [f for f in _RAG_INDEX if f["path"] not in paths]
    else:
        new_index = []
        
    base = MCP_ROOT if MCP_ROOT else _BASE_DIR
    for rel in paths:
        try:
            abs_p = (base / rel).resolve()
            if not str(abs_p).startswith(str(base)): continue
            # Supprimer de l'index si fichier supprimé
            if not abs_p.is_file(): 
                continue
            
            # Ignorer dossiers techniques
            parts_low = [pt.lower() for pt in abs_p.parts]
            if any(pt.startswith('.') for pt in abs_p.parts) or "node_modules" in parts_low or "__pycache__" in parts_low:
                continue

            content = abs_p.read_text(encoding="utf-8", errors="replace")
            # Chunking simple
            chunks_raw = [content[i:i+1500] for i in range(0, len(content), 1000)]
            chunks = []
            for c in chunks_raw:
                chunks.append({"text": c, "tokens": tokenize(c)})
            new_index.append({"path": rel, "chunks": chunks})
        except: pass
    
    _RAG_INDEX = new_index
    return {"ok": True, "files": len(_RAG_INDEX), "chunks": sum(len(f["chunks"]) for f in _RAG_INDEX)}

async def run_project_watcher():
    """Tâche de fond surveillant les changements de fichiers dans MCP_ROOT."""
    import asyncio
    from watchfiles import awatch, Change
    
    last_root = None
    while True:
        try:
            current_root = MCP_ROOT
            if not current_root:
                await asyncio.sleep(5)
                continue
            
            print(f"[WATCHER] Surveillance de : {current_root}")
            # awatch surveille récursivement par défaut
            async for changes in awatch(current_root):
                files_to_update = []
                for change_type, path_str in changes:
                    p = pathlib.Path(path_str)
                    try:
                        rel = str(p.relative_to(current_root))
                        # Ignorer les fichiers temporaires/système
                        if any(pt.startswith('.') for pt in p.parts) or "node_modules" in p.parts or "__pycache__" in p.parts:
                            continue
                        files_to_update.append(rel)
                    except ValueError:
                        continue
                
                if files_to_update:
                    print(f"[WATCHER] Update RAG pour : {files_to_update}")
                    await rag_index_internal(files_to_update, partial=True)
                
                # Petite pause pour ne pas saturer si modifs massives
                await asyncio.sleep(0.5)

            # Si on sort de la boucle awatch (ex: root changé)
            await asyncio.sleep(2)
        except Exception as e:
            print(f"[WATCHER] Erreur: {e}")
            await asyncio.sleep(5)

@app.post("/api/rag/index")
async def rag_index_files_route(req: Request):
    data = await req.json()
    paths = data.get("paths", [])
    res = await rag_index_internal(paths)
    return res

def rag_search_internal(q: str, limit: int = 5):
    if not _RAG_INDEX: return {"results": []}
    
    all_chunks: List[Dict[str, Any]] = []
    for f in _RAG_INDEX:
        for c in f.get("chunks", []):
            all_chunks.append({"path": f.get("path"), "text": c.get("text"), "tokens": c.get("tokens")})
    
    if not all_chunks: return {"results": []}
    
    n_chunks = len(all_chunks)
    avg_len = sum(sum(c["tokens"].values()) for c in all_chunks) / n_chunks
    query_tokens = list(tokenize(q).keys())
    
    token_counts = Counter()
    for c in all_chunks:
        for t in c["tokens"]: 
            token_counts[t] += 1
            
    scored = []
    for c in all_chunks:
        score = 0.0
        for t in query_tokens:
            if t in c["tokens"]:
                n_containing = token_counts[t]
                score += b25_score([t], c["tokens"], avg_len, n_chunks, n_containing)
        scored.append({"path": str(c["path"]), "text": str(c["text"]), "score": score})
    
    scored.sort(key=lambda x: x["score"], reverse=True)
    return {"results": scored[:limit]}

@app.get("/api/rag/search")
def rag_search_route(q: str, limit: int = 5):
    return rag_search_internal(q, limit)

@app.get("/api/debug/models")
async def debug_models(q: str = ""):
    """Liste les IDs OpenRouter contenant le mot-cle q."""
    cfg = load_config()
    api_key = cfg.get("api_key", "")
    try:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(f"{BASE_URL}/models", headers=headers)
            r.raise_for_status()
            data = r.json().get("data", [])
            ids = sorted([m["id"] for m in data if not q or q.lower() in m["id"].lower()])
            return {"count": len(ids), "ids": ids}
    except Exception as e:
        return {"error": str(e)}

@app.get("/favicon.ico")
async def favicon():
    # Retourne un SVG minimaliste comme favicon
    svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><text y=".9em" font-size="90">✦</text></svg>'
    from fastapi.responses import Response
    return Response(content=svg, media_type="image/svg+xml")

@app.get("/api/proxy_image")
async def proxy_image(req: Request):
    import urllib.parse
    import re
    import traceback
    
    try:
        # 1. Récupération dynamique (zéro risque d'erreur d'import comme Optional)
        prompt = req.query_params.get("prompt")
        url_param = req.query_params.get("url")
        seed = req.query_params.get("seed", "42")
        model = req.query_params.get("model", "flux")
        
        # 2. Construction de l'URL ultra-propre
        if prompt:
            prompt_quoted = urllib.parse.quote(prompt)
            clean_url = f"https://image.pollinations.ai/prompt/{prompt_quoted}?width=1024&height=1024&seed={seed}&model={model}&nologo=true"
        elif url_param:
            clean_url = urllib.parse.unquote(url_param)
            clean_url = re.sub(r"&key=[^&]*", "", clean_url)
        else:
            return JSONResponse({"error": "Paramètres manquants"}, status_code=400)

        print(f"[PROXY] Téléchargement de l'image ({model})...")

        # 3. Appel à Pollinations
        async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
            }
            resp = await client.get(clean_url, headers=headers)
            
            # 4. Détection du blocage ou contenu invalide
            content_type = resp.headers.get("content-type", "image/png").split(";")[0].strip().lower()
            content_len = len(resp.content)
            
            if "text/html" in content_type or "application/json" in content_type or content_len < 1000:
                 err_body = resp.text[:200]
                 print(f"[PROXY ERROR] Bloqué ou invalide : {content_type} ({content_len} octets)")
                 print(f"[API MSG] {err_body}")
                 return JSONResponse({"error": f"Pollinations error: {err_body}"}, status_code=403)
            
            print(f"[PROXY] Succès : {content_type} ({content_len} octets)")
            from fastapi.responses import StreamingResponse
            return StreamingResponse(io.BytesIO(resp.content), media_type=content_type)
            
    except Exception as e:
        # 5. Affichage brutal de l'erreur dans ton terminal noir si ça plante
        print(f"\n[PROXY ERREUR FATALE]")
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

# ─────────────────────────────────────────────
#  BATTLE MODE — Ultra-Smooth Character-Stream
# ─────────────────────────────────────────────
@app.post("/api/battle")
async def battle_endpoint(req: Request):
    """Streaming caractère par caractère pour une fluidité absolue."""
    body = await req.json()
    cfg = load_config()
    api_key = cfg.get("api_key", "")
    model_left = body.get("model_left")
    model_right = body.get("model_right")
    messages = body.get("messages", [])
    temperature = body.get("temperature", 0.7)
    
    api_messages = [{"role": "system", "content": "Assistant précis."}] + trim_history(messages)

    queues = {"left": asyncio.Queue(), "right": asyncio.Queue()}
    status = {"left": "active", "right": "active"}

    async def stream_task(model, side):
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream("POST", f"{BASE_URL}/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}", **OPENROUTER_HEADERS},
                    json={"model": model, "messages": api_messages, "temperature": temperature, "stream": True}
                ) as resp:
                    if resp.status_code != 200:
                        await queues[side].put(f"⚠️ Erreur {resp.status_code}")
                    else:
                        async for line in resp.aiter_lines():
                            if not line.startswith("data: "): continue
                            line = line[6:].strip()
                            if line == "[DONE]": break
                            try:
                                d = json.loads(line)
                                delta = d.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                if delta:
                                    # ON ENVOIE CHAQUE CARACTÈRE INDIVIDUELLEMENT
                                    for char in delta:
                                        await queues[side].put(char)
                            except: pass
        except Exception as e:
            await queues[side].put(f"❌ Erreur: {str(e)}")
        finally:
            status[side] = "done"

    async def battle_generator():
        asyncio.create_task(stream_task(model_left, "left"))
        asyncio.create_task(stream_task(model_right, "right"))

        while status["left"] == "active" or status["right"] == "active" or not queues["left"].empty() or not queues["right"].empty():
            has_data = False
            for side in ["left", "right"]:
                if not queues[side].empty():
                    char = await queues[side].get()
                    yield f"data: {json.dumps({'side': side, 'delta': char, 'done': False})}\n\n"
                    has_data = True
            
            # Pause micro pour la fluidité
            await asyncio.sleep(0.002 if has_data else 0.001)

        yield f"data: {json.dumps({'side': 'left', 'done': True})}\n\n"
        yield f"data: {json.dumps({'side': 'right', 'done': True})}\n\n"
        yield "data: {\"battle_done\": true}\n\n"

    from fastapi.responses import StreamingResponse as _SR
    return _SR(battle_generator(), media_type="text/event-stream")


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    with open(os.path.join(_BASE_DIR, "index.html"), "r", encoding="utf-8") as f:
        return f.read()
