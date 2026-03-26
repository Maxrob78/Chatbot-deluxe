import json
import os
import base64
import io
from datetime import datetime
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from pdfminer.high_level import extract_text as pdf_extract_text
import httpx
import re
from ddgs import DDGS
import trafilatura
import subprocess
import pathlib
import math
from collections import Counter
from typing import List, Dict, Any, Optional

# ─────────────────────────────────────────────
#  HELPERS — DATE
# ─────────────────────────────────────────────
def get_current_date_info() -> str:
    """Retourne la date et l'heure actuelle en français pour le contexte de l'IA."""
    now = datetime.now()
    days = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
    months = ["Janvier", "Février", "Mars", "Avril", "Mai", "Juin", "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre"]
    return f"Nous sommes actuellement le {days[now.weekday()]} {now.day} {months[now.month-1]} {now.year}. Heure : {now.strftime('%H:%M')}."

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
# Chemins absolus relatifs au dossier de main.py
_BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE = os.path.join(_BASE_DIR, "history.json")
CONFIG_FILE  = os.path.join(_BASE_DIR, "config.json")
MEMORY_FILE  = os.path.join(_BASE_DIR, "memory.json")
BASE_URL        = "https://openrouter.ai/api/v1"
MAX_HISTORY     = 100
MAX_FILE_MB     = 10
REQUEST_TIMEOUT = 60

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    """Precharge les scores Artificial Analysis au demarrage."""
    try:
        cfg = load_config()
        await load_aa_scores(cfg.get("aa_api_key", ""))
    except Exception as e:
        print(f"[STARTUP] Erreur AA (non bloquant): {e}")


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
        try: return json.load(f)
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
            total += len(text_content) // 4
            
    return total

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
        "3. Ne te laisse pas distraire par l'actualité brûlante si des dates officielles (calendrier) existent ailleurs. "
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
                    cost = float(p.get("completion", 0)) * 1_000_000
                    if cost == 0: return "Gratuit"
                    return f"${cost:.2f}/M" if cost < 1 else f"${cost:.1f}/M"
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

async def fetch_page_content(url: str, max_chars: int = 2500) -> str:
    """Récupère et nettoie proprement le texte d'une page web via trafilatura."""
    try:
        # 1. Sécurité : Ignorer les URLs non-HTTP ou fichiers lourds/binaires
        if not url.startswith(("http://", "https://")):
            return ""
        if any(url.lower().endswith(ext) for ext in [".pdf", ".jpg", ".png", ".gif", ".zip", ".mp4", ".tar", ".gz"]):
            return ""

        # 2. Télécharger la page via httpx
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
        }
        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
            r = await client.get(url, headers=headers)
            if r.status_code != 200: 
                return ""
            html = r.text

        # 3. Extraction intelligente
        content = trafilatura.extract(html, include_comments=False, no_fallback=False)
        
        if not content: 
            return ""
        # Nettoyer un peu plus pour le LLM
        lines = [line.strip() for line in content.split("\n") if len(line.strip()) > 15]
        content = "\n".join(lines)
        return content[:max_chars].strip()
    except Exception as e:
        print(f"[SEARCH] Erreur fetch_page ({url}): {e}")
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
    r"PDG|CEO|président|premier ministre|ministre|directeur|qui dirige|qui est le patron|"
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

    # ── 1. PRIORITÉ ABSOLUE : Les demandes explicites ou hyper-actuelles ──
    if RE_EXPLICIT_SEARCH.search(low):
        return text

    # ── 2. ENRICHISSEMENT AUTO (Sport, PSG, Calendrier) ──
    if "psg" in low and any(k in low for k in ["match", "adversaire", "calendrier", "joue", "prochain"]):
        return text + " calendrier officiel complet Ligue 1 2026"

    # ── 3. SUJETS DYNAMIQUES (Stats, Prix, Météo, etc.) ──
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
        return ""
    date_info = get_current_date_info()
    lines = [f"[Recherche web — {query}]", f"[{date_info}]", ""]
    for i, r in enumerate(results, 1):
        lines.append(f"## {r.get('title','Source ' + str(i))}")
        if r.get('href'): lines.append(f"LIEN: {r['href']}")
        
        body = r.get("content") or r.get("snippet","")
        if body:
            # On réduit un peu la taille pour le contexte
            lines.append(body[:1200])
        lines.append("")
        
    lines.append("[INSTRUCTION : Analyse ces résultats. Réponds directement si tu as l'info. Si tu fais de nouvelles recherches, sois plus spécifique.]")
    lines.append("[Note : Citations Markdown cliquables requises.]")
    return "\n".join(lines)

@app.get("/api/search")
async def search_endpoint(q: str = ""):
    if not q:
        return {"results": []}
    results = await ddg_search(q)
    return {"results": results, "query": q}

# ─────────────────────────────────────────────
#  DEVELOPER TOOLS LOGIC (Aider-style & Git)
# ─────────────────────────────────────────────
def apply_aider_diff(content: str, patch: str) -> str:
    """Applique un patch au format SEARCH/REPLACE (Aider style)."""
    import re
    # Extraction des blocs SEARCH/REPLACE
    # Format: <<<<<<< SEARCH\n(vieux code)\n=======\n(nouveau code)\n>>>>>>> REPLACE
    pattern = r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE"
    blocks = re.findall(pattern, patch, re.DOTALL)
    if not blocks:
        return content

    new_content = content
    for search, replace in blocks:
        if search in new_content:
            new_content = new_content.replace(search, replace, 1)
        else:
            # Tentative de match plus souple (sans espaces de fin si besoin)
            s_clean = search.strip()
            if s_clean in new_content:
                new_content = new_content.replace(s_clean, replace, 1)
            else:
                raise Exception(f"Bloc SEARCH introuvable dans le fichier : {search[:100]}...")
    return new_content

AGENT_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Recherche des informations récentes sur le web via DuckDuckGo.",
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
]

async def execute_tool(name: str, args: dict) -> str:
    """Exécute un outil et retourne le résultat sous forme de string."""
    try:
        if name == "web_search":
            results = await ddg_search(args.get("query",""), max_results=4, fetch_content=True)
            if not results: return "Aucun résultat trouvé."
            return build_search_context(results, args.get("query",""))

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
            return f"Fichier écrit avec succès : {path} ({len(content.encode())} bytes)"

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
            # Analyse statique de sécurité
            forbidden = {"os", "subprocess", "shutil", "sys", "pathlib", "socket", "pty"}
            try:
                tree = ast.parse(code)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if alias.name.split('.')[0] in forbidden:
                                return f"Erreur de sécurité : L'import du module '{alias.name}' est interdit."
                    elif isinstance(node, ast.ImportFrom):
                        if node.module and node.module.split('.')[0] in forbidden:
                            return f"Erreur de sécurité : L'import depuis '{node.module}' est interdit."
                    elif isinstance(node, ast.Call):
                        if isinstance(node.func, ast.Name) and node.func.id in ['__import__', 'eval', 'exec', 'open']:
                            return f"Erreur de sécurité : L'utilisation de la fonction '{node.func.id}()' est interdite."
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
                return f"Fichier {path} modifié avec succès (Diff appliqué)."
            except Exception as e:
                return f"Erreur application Diff : {str(e)}"

        elif name == "run_git":
            if not MCP_ROOT: return "Erreur : aucun dossier MCP configuré."
            git_args = args.get("args","status")
            import subprocess
            try:
                # On lance Git dans le dossier MCP_ROOT
                res = subprocess.run(
                    f"git {git_args}", 
                    cwd=str(MCP_ROOT), 
                    shell=True, capture_output=True, text=True, encoding="utf-8"
                )
                out = res.stdout if res.stdout else ""
                err = res.stderr if res.stderr else ""
                return (out + "\n" + err).strip() or "(commande exécutée, pas de sortie)"
            except Exception as e:
                return f"Erreur Git : {str(e)}"
                        
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

        elif name == "run_shell":
            import subprocess
            command = args.get("command", "")
            if not command: return "Erreur : commande vide."
            try:
                # Exécuter dans le dossier MCP racine si défini, sinon dossier du script
                cwd = MCP_ROOT if MCP_ROOT else _BASE_DIR
                # Note: shell=True permet d'utiliser les pipes, redirections, etc.
                process = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    cwd=cwd,
                    timeout=60
                )
                output = process.stdout
                if process.stderr:
                    output += f"\n[stderr]\n{process.stderr}"
                return output if output else "(commande exécutée sans sortie console)"
            except subprocess.TimeoutExpired:
                return "Erreur : La commande a expiré (timeout 60s)."
            except Exception as e:
                return f"Erreur Shell : {e}"

        else:
            return f"Outil inconnu : {name}"
    except Exception as e:
        return f"Erreur lors de l'exécution de {name} : {e}"

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
    system_prompt = cfg.get("system_prompt","Tu es un assistant utile, précis et concis.")

    # Injecter mémoire et date
    date_info    = get_current_date_info()
    memory_items = load_memory()
    full_system  = f"{date_info}\n\n{system_prompt.strip()}"
    if memory_items:
        mem_text = "\n".join(f"- {m['text']}" for m in memory_items if m.get("text","").strip())
        if mem_text:
            full_system += f"\n\n[Mémoire persistante :]\n{mem_text}"

    # Contexte MCP
    if MCP_ROOT:
        full_system += f"\n\n[MCP Filesystem actif — racine : {MCP_ROOT}]\nTu peux lire/écrire des fichiers via les outils disponibles."

    agent_messages = [{"role":"system","content":full_system}] + messages
    max_iterations = 15

    async def agent_stream():
        nonlocal agent_messages
        iteration = 0

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
                        req_body["tool_choice"] = "auto"
                        req_body["stream"] = False
                    else:
                        # Fallback : streaming natif sans tools
                        req_body["stream"] = True

                    if use_tools:
                        # Non-streaming pour function calling
                        resp = await client.post(
                            f"{BASE_URL}/chat/completions",
                            headers={"Authorization": f"Bearer {api_key}"},
                            json=req_body
                        )
                        if resp.status_code == 400:
                            err_body = resp.json()
                            err_msg  = str(err_body.get("error",""))
                            # Si erreur liée aux tools → basculer en fallback
                            if any(k in err_msg.lower() for k in ["tool","function","unsupported","not support"]):
                                print(f"[AGENT] Modèle sans function calling, fallback texte")
                                use_tools = False
                                yield f"data: {json.dumps({'info': 'Modèle sans function calling, mode texte'})}\n\n"
                                iteration -= 1
                                continue
                            yield f"data: {json.dumps({'error': err_msg})}\n\n"
                            return
                        elif resp.status_code >= 400:
                            err = resp.json().get("error",{})
                            msg = err.get("message",str(err)) if isinstance(err,dict) else str(err)
                            yield f"data: {json.dumps({'error': msg})}\n\n"
                            return
                        result  = resp.json()
                        choice  = result["choices"][0]
                        message = choice["message"]
                        reason  = choice.get("finish_reason","")
                        agent_messages.append(message)

                        # Cas 1 : appel d'outils
                        if reason == "tool_calls" or message.get("tool_calls"):
                            tool_calls = message.get("tool_calls",[])
                            for tc in tool_calls:
                                fn_name = tc["function"]["name"]
                                try:
                                    fn_args = json.loads(tc["function"].get("arguments","{}"))
                                except:
                                    fn_args = {}
                                yield f"data: {json.dumps({'tool_call': True, 'tool': fn_name, 'args': fn_args})}\n\n"
                                tool_result = await execute_tool(fn_name, fn_args)
                                print(f"[AGENT] {fn_name} → {len(tool_result)} chars")
                                yield f"data: {json.dumps({'tool_result': True, 'tool': fn_name, 'preview': tool_result[:100]})}\n\n"
                                agent_messages.append({
                                    "role": "tool",
                                    "tool_call_id": tc["id"],
                                    "content": tool_result[:8000]
                                })
                            continue  # Prochain tour de boucle

                        # Cas 2 : réponse finale (non-streaming) → on la stream mot par mot
                        final_text = message.get("content","")
                        if final_text:
                            import asyncio
                            words = final_text.split(" ")
                            for i in range(0, len(words), 4):
                                chunk = " ".join(words[i:i+4])
                                if i + 4 < len(words): 
                                    chunk = str(chunk) + " "
                                yield f"data: {json.dumps({'delta': chunk})}\n\n"
                                await asyncio.sleep(0.008)
                            messages.append({"role":"assistant","content":final_text})
                            upsert_conversation(session_id, model, messages)
                            yield f"data: {json.dumps({'done': True, 'tokens': estimate_tokens(messages)})}\n\n"
                            return

                    else:
                        # Fallback : mode texte avec instructions dans le system prompt
                        # Injecter les instructions d'outils dans le dernier message système
                        fallback_system = agent_messages[0]["content"] if agent_messages else ""
                        fallback_system += (
                            "\n\nTu es en mode agent. Pour utiliser un outil, ecris EXACTEMENT:\n"
                            "TOOL:web_search:<requete>\n"
                            "TOOL:read_file:<chemin>\n"
                            "TOOL:write_file:<chemin>\n<contenu>\n"
                            "TOOL:list_files:<dossier>\n"
                            "Sinon reponds normalement."
                        )
                        fb_messages = [{"role":"system","content":fallback_system}] + agent_messages[1:]

                        # Streaming natif
                        full_text = ""
                        async with client.stream(
                            "POST", f"{BASE_URL}/chat/completions",
                            headers={"Authorization": f"Bearer {api_key}"},
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
                                        yield f"data: {json.dumps({'delta': delta})}\n\n"
                                except: pass

                        # Détecter appel d'outil dans la réponse texte
                        import re as _re3
                        # Détecter appel d'outil dans la réponse texte
                        import re as _re3
                        tool_match = _re3.search(r"TOOL:(\w+):(.+?)(?:\n|$)", full_text, _re3.S)
                        if tool_match and iteration < max_iterations:
                            fn_name = tool_match.group(1)
                            fn_arg  = tool_match.group(2).strip()
                            fn_args = {"query": fn_arg} if fn_name == "web_search" else \
                                      {"path": fn_arg.split("\n")[0], "content": "\n".join(fn_arg.split("\n")[1:])} if fn_name == "write_file" else \
                                      {"path": fn_arg}
                            yield f"data: {json.dumps({'tool_call': True, 'tool': fn_name, 'args': fn_args})}\n\n"
                            tool_result = await execute_tool(fn_name, fn_args)
                            yield f"data: {json.dumps({'tool_result': True, 'tool': fn_name, 'preview': tool_result[:100]})}\n\n"
                            agent_messages.append({"role":"assistant","content":full_text})
                            agent_messages.append({"role":"user","content":"[Resultat "+fn_name+"] :\n"+tool_result[:6000]})
                            continue

                        # Réponse finale en fallback
                        messages.append({"role":"assistant","content":full_text})
                        upsert_conversation(session_id, model, messages)
                        yield f"data: {json.dumps({'done': True, 'tokens': estimate_tokens(messages)})}\n\n"
                        return

            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                return

        yield f"data: {json.dumps({'error': 'Limite de '+str(max_iterations)+' iterations atteinte'})}\n\n"
    return StreamingResponse(agent_stream(), media_type="text/event-stream")

# ─────────────────────────────────────────────
#  ROUTES — BALANCE
# ─────────────────────────────────────────────
@app.get("/api/balance")
async def get_balance():
    cfg = load_config()
    api_key = cfg.get("api_key", "")
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"{BASE_URL}/auth/key", headers={"Authorization": f"Bearer {api_key}"})
            r.raise_for_status()
            data = r.json().get("data", {})
            return {"usage": f"{data.get('usage', 0):.4f} $"}
    except Exception as e:
        return {"error": str(e)}

# ─────────────────────────────────────────────
#  ROUTES — HISTORY
# ─────────────────────────────────────────────
@app.get("/api/history")
def get_history():
    return {"history": load_history()}

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
    model          = body.get("model")
    messages       = body.get("messages", [])
    temperature    = body.get("temperature", 0.7)
    max_tokens     = body.get("max_tokens", 2048)
    system_prompt  = cfg.get("system_prompt", "Tu es un assistant utile, précis et concis.")
    web_search_on  = body.get("web_search", True)

    def sanitize_messages(msgs):
        """Nettoie les messages avant envoi à OpenRouter."""
        clean = []
        for m in msgs:
            role = m.get("role")
            c    = m.get("content")
            if not c:
                continue
            if isinstance(c, list):
                blocks = []
                has_image = False
                for b in c:
                    if b.get("type") == "text" and b.get("text", "").strip():
                        blocks.append({"type": "text", "text": b["text"].strip()})
                    elif b.get("type") == "image_url":
                        url = b.get("image_url", {}).get("url", "")
                        if url.startswith("data:image/") or url.startswith("http"):
                            blocks.append(b)
                            has_image = True
                if not blocks:
                    continue
                # Si pas d'image : envoyer comme string simple (compatible tous modèles)
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
    if memory_items:
        mem_text = "\n".join(f"- {m['text']}" for m in memory_items if m.get("text","").strip())
        if mem_text:
            full_system += f"\n\n[Mémoire persistante — informations sur l\'utilisateur :]\n{mem_text}"

    # Recherche web automatique si la question le nécessite
    search_query = should_search(messages) if web_search_on else None
    search_results_text = ""
    if search_query:
        results = await ddg_search(search_query, max_results=10)
        if results:
            search_results_text = build_search_context(results, search_query)
            full_system += f"\n\n{search_results_text}"
            print(f"[SEARCH] {len(results)} résultats pour: {search_query[:60]}")

    # RAG local automatique si indexé
    if _RAG_INDEX and messages:
        last_user_msg = ""
        last = messages[-1]
        if last.get("role") == "user":
            c = last.get("content", "")
            last_user_msg = " ".join(b.get("text", "") for b in c if b.get("type") == "text") if isinstance(c, list) else str(c)
        
        if last_user_msg:
            rag_res = rag_search(last_user_msg, limit=4)
            if rag_res and rag_res.get("results"):
                rag_context = "\n\n[Contexte Local (RAG) — fichiers du projet :]\n"
                for r in rag_res["results"]:
                    if r["score"] > 0:
                        rag_context += f"### FICHIER: {r['path']}\n{r['text']}\n---\n"
                full_system += rag_context

    messages_to_send = []
    if full_system:
        messages_to_send.append({"role": "system", "content": full_system})
    messages_to_send.extend(sanitize_messages(messages))

    async def stream_generator():
        full_res = ""
        # Signaler au frontend si une recherche a été faite
        if search_results_text:
            yield f"data: {json.dumps({'searching': True, 'query': search_query[:80]})}\n\n"
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                async with client.stream(
                    "POST",
                    f"{BASE_URL}/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={"model": model, "messages": messages_to_send,
                          "stream": True, "temperature": temperature, "max_tokens": max_tokens},
                ) as r:
                    # Lire le body d'erreur avant raise pour avoir le vrai message
                    if r.status_code >= 400:
                        err_body = await r.aread()
                        try:
                            err_json = json.loads(err_body)
                            err_msg  = err_json.get("error", {})
                            if isinstance(err_msg, dict):
                                err_msg = err_msg.get("message", str(err_json))
                            else:
                                err_msg = str(err_msg)
                        except:
                            err_msg = err_body.decode(errors="replace")[:300]
                        yield f"data: {json.dumps({'error': f'Erreur {r.status_code} : {err_msg}'})}\n\n"
                        return
                    async for line in r.aiter_lines():
                        if not line or not line.startswith("data: "): continue
                        payload = line[6:]
                        if payload == "[DONE]": break
                        try:
                            chunk = json.loads(payload)
                            delta = chunk["choices"][0].get("delta", {}).get("content", "")
                            if isinstance(delta, str) and delta:
                                # Cast explicite pour le linter
                                full_res = str(full_res) + delta
                                yield f"data: {json.dumps({'delta': delta})}\n\n"
                        except: pass
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            if full_res:
                messages.append({"role": "assistant", "content": full_res})
                upsert_conversation(session_id, model, messages)
                tokens = estimate_tokens(messages)
            else:
                tokens = estimate_tokens(messages)
            yield f"data: {json.dumps({'done': True, 'tokens': tokens})}\n\n"

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

# ─────────────────────────────────────────────
#  GIT INTEGRATION
# ─────────────────────────────────────────────
def run_git(args: list) -> dict:
    try:
        res = subprocess.run(["git"] + args, capture_output=True, text=True, cwd=_BASE_DIR, encoding="utf-8", errors="replace")
        return {"ok": res.returncode == 0, "stdout": res.stdout, "stderr": res.stderr, "code": res.returncode}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/api/git/status")
def git_status():
    if not os.path.exists(os.path.join(_BASE_DIR, ".git")):
        return {"ok": False, "not_repo": True}
    res = run_git(["status", "--porcelain"])
    branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    return {
        "ok": res["ok"], 
        "status": res["stdout"], 
        "branch": branch["stdout"].strip() if branch["ok"] else "???"
    }

@app.post("/api/git/init")
def git_init():
    res = run_git(["init"])
    if res["ok"]:
        # Premier commit auto ?
        run_git(["add", "."])
        run_git(["commit", "-m", "Initial commit via Chatbot Deluxe"])
    return res

@app.post("/api/git/commit")
async def git_commit(req: Request):
    data = await req.json()
    msg = data.get("message", "Update via Chatbot")
    run_git(["add", "."])
    return run_git(["commit", "-m", msg])

@app.get("/api/git/diff")
def git_diff():
    return run_git(["diff", "HEAD"])

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
        abs_path = os.path.join(_BASE_DIR, rel_path)
        
        if not os.path.exists(abs_path):
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

def b25_score(query_tokens, chunk_tokens, avg_len, n_chunks, n_containing):
    # Version très simplifiée de BM25
    score = 0
    k1 = 1.5
    b = 0.75
    for token in query_tokens:
        if token not in chunk_tokens: continue
        f = chunk_tokens[token]
        # IDF
        idf = math.log((n_chunks - n_containing + 0.5) / (n_containing + 0.5) + 1.0)
        # Term Frequency saturation
        c_len = sum(chunk_tokens.values())
        tf = (f * (k1 + 1)) / (f + k1 * (1 - b + b * (c_len / avg_len)))
        score += idf * tf
    return score

@app.post("/api/rag/index")
async def rag_index_files(req: Request):
    global _RAG_INDEX
    data = await req.json()
    paths = data.get("paths", []) # list of rel paths
    
    new_index = []
    for rel in paths:
        abs_p = os.path.join(_BASE_DIR, rel)
        if not os.path.exists(abs_p) or not os.path.isfile(abs_p): continue
        
        try:
            content = ""
            with open(abs_p, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            
            # Chunking simple (par paragraphes ou 1000 chars)
            chunks_raw = [content[i:i+1500] for i in range(0, len(content), 1000)]
            chunks = []
            for c in chunks_raw:
                chunks.append({"text": c, "tokens": tokenize(c)})
            new_index.append({"path": rel, "chunks": chunks})
        except: pass
    
    _RAG_INDEX = new_index
    return {"ok": True, "files": len(_RAG_INDEX), "chunks": sum(len(f["chunks"]) for f in _RAG_INDEX)}

@app.get("/api/rag/search")
def rag_search(q: str, limit: int = 5):
    if not _RAG_INDEX: return {"results": []}
    
    all_chunks: List[Dict[str, Any]] = []
    for f in _RAG_INDEX:
        for c in f.get("chunks", []):
            all_chunks.append({"path": f.get("path"), "text": c.get("text"), "tokens": c.get("tokens")})
    
    if not all_chunks: return {"results": []}
    
    n_chunks = len(all_chunks)
    avg_len = sum(sum(c["tokens"].values()) for c in all_chunks) / n_chunks
    query_tokens = tokenize(q)
    
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
                # cast explicite pour eviter les erreurs de types a l indexation
                score += float(b25_score([t], c["tokens"], avg_len, n_chunks, n_containing))
        scored.append({"path": str(c["path"]), "text": str(c["text"]), "score": score})
    
    scored.sort(key=lambda x: x["score"], reverse=True)
    return {"results": scored[:limit]}

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

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    with open(os.path.join(_BASE_DIR, "index.html"), "r", encoding="utf-8") as f:
        return f.read()
