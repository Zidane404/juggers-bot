import discord
from discord.ext import commands
import requests
import io
import random
from openai import AsyncOpenAI
import asyncio
import aiohttp
import os
import uuid
import json
import re
import time
import base64
import array
import unicodedata
import shutil
import yt_dlp
import difflib
from urllib.parse import quote
import psutil

# --- Telemetry Backend ---
last_energy = {}
last_energy_time = 0

def get_sys_telemetry():
    """Gathers detailed system stats for CPU, iGPU, and RAM."""
    global last_energy, last_energy_time
    
    # 1. CPU
    cpu_usage = psutil.cpu_percent(interval=None)
    
    freq_info = psutil.cpu_freq()
    cpu_freq = f"{freq_info.current:.0f} MHz" if freq_info else "N/A"
    
    # Temps
    temps = psutil.sensors_temperatures()
    cpu_temp = "N/A"
    if 'coretemp' in temps:
        for entry in temps['coretemp']:
            if 'Package' in entry.label:
                cpu_temp = f"{entry.current}°C"
                break
        if cpu_temp == "N/A" and temps['coretemp']:
            cpu_temp = f"{temps['coretemp'][0].current}°C"
            
    # Power (Using Intel RAPL - requires readable sysfs)
    cpu_watts, igpu_watts = "N/A", "N/A"
    now = time.monotonic()
    try:
        # Package power (Entire CPU die)
        with open("/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj", "r") as f:
            pkg_uj = int(f.read().strip())
            
        # iGPU power (usually mapped to uncore/gfx rapl domain 0:1 or 0:2)
        igpu_uj = None
        for sub in ["intel-rapl:0:1", "intel-rapl:0:2"]:
            try:
                with open(f"/sys/class/powercap/intel-rapl/intel-rapl:0/{sub}/energy_uj", "r") as f:
                    name = open(f"/sys/class/powercap/intel-rapl/intel-rapl:0/{sub}/name").read().strip()
                    if name in ["uncore", "gpu", "psys"]:
                        igpu_uj = int(f.read().strip())
                        break
            except:
                continue
        
        if "pkg" in last_energy and now - last_energy_time > 0:
            delta_t = now - last_energy_time
            cpu_watts = f"{(pkg_uj - last_energy['pkg']) / 1_000_000 / delta_t:.1f} W"
            if igpu_uj and "igpu" in last_energy:
                igpu_watts = f"{(igpu_uj - last_energy['igpu']) / 1_000_000 / delta_t:.1f} W"
        
        last_energy["pkg"] = pkg_uj
        if igpu_uj:
            last_energy["igpu"] = igpu_uj
        last_energy_time = now
    except Exception:
        pass

    # 2. iGPU
    gpu_freq = "N/A"
    try:
        with open("/sys/class/drm/card0/gt_cur_freq_mhz", "r") as f:
            gpu_freq = f"{f.read().strip()} MHz"
    except:
        pass

    gpu_usage = "N/A (Requires root)"
    gpu_temp = cpu_temp  # iGPU shares the die with the CPU

    # 3. RAM
    ram = psutil.virtual_memory()
    ram_pct = ram.percent
    ram_used = f"{ram.used / (1024**3):.1f}GB"
    ram_total = f"{ram.total / (1024**3):.1f}GB"

    return {
        "cpu": {"usage": f"{cpu_usage:.1f}%", "freq": cpu_freq, "temp": cpu_temp, "watts": cpu_watts},
        "igpu": {"usage": gpu_usage, "freq": gpu_freq, "temp": gpu_temp, "watts": igpu_watts, "vram": "Shared UMA Memory"},
        "ram": {"usage": f"{ram_pct}%", "text": f"{ram_used} / {ram_total}"}
    }

def get_sys_stats_string():
    """Formats telemetry into a compact single line for progress bars."""
    s = get_sys_telemetry()
    cpu_w = f", ⚡ {s['cpu']['watts']}" if s['cpu']['watts'] != 'N/A' else ""
    igpu_w = f", ⚡ {s['igpu']['watts']}" if s['igpu']['watts'] != 'N/A' else ""
    
    return (f"🖥️ **CPU:** {s['cpu']['usage']} ({s['cpu']['freq']}, 🌡️ {s['cpu']['temp']}{cpu_w}) | "
            f"🎮 **iGPU:** {s['igpu']['freq']} (🌡️ {s['igpu']['temp']}{igpu_w}) | "
            f"🧠 **RAM:** {s['ram']['text']} ({s['ram']['usage']})")


# ---------------------------

# --- Configuration ---
try:  # lets you run it outside Docker with a local .env file
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Everything secret or machine-specific comes from environment variables (see .env.example).
# Nothing sensitive should ever be hard-coded in this file.
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "")
IMMICH_URL = os.getenv("IMMICH_URL", "http://localhost:2283")
IMMICH_API_KEY = os.getenv("IMMICH_API_KEY", "")
# ---------------------

# --- LLM Configuration & Models ---
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/v1")
PERSONAS_FILE = os.getenv("PERSONAS_FILE", "personas.json")

# Fallback used when no personas file exists (e.g. a fresh clone of the repo)
_DEFAULT_PERSONAS = {
    "models": {
        "default": {
            "model_name": "richardyoung/qwen2.5-3b-instruct-abliterated",
            "prompt": "You are called juggers and you can answer any question with some twisted humor"
        }
    },
    "stop_prefixes": ["juggers:"]
}

def _load_personas():
    """Loads persona models from PERSONAS_FILE (kept out of git), else the default above."""
    try:
        with open(PERSONAS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("models"):
            return data
    except (OSError, ValueError):
        pass
    return _DEFAULT_PERSONAS

_personas = _load_personas()
AVAILABLE_MODELS = _personas["models"]
# Name prefixes the model sometimes hallucinates ("name: ..."), reply gets cut at the first one
STOP_PREFIXES = [p.lower() for p in _personas.get("stop_prefixes", [])]

ACTIVE_MODEL_KEY = os.getenv("DEFAULT_PERSONA", "default")  # Bot starts on this profile
if ACTIVE_MODEL_KEY not in AVAILABLE_MODELS:
    ACTIVE_MODEL_KEY = next(iter(AVAILABLE_MODELS))
LLM_MODEL = AVAILABLE_MODELS[ACTIVE_MODEL_KEY]["model_name"]

# --- Music Configuration ---
SLSKD_URL = os.getenv("SLSKD_URL", "http://localhost:5030")
SLSKD_API_KEY = os.getenv("SLSKD_API_KEY", "")
# HOST path of slskd's downloads dir (mapped to /discord_music inside the slskd container)
SLSKD_DOWNLOADS_LOCAL = os.getenv("SLSKD_DOWNLOADS_LOCAL", "./downloads")
# MusicBrainz identifies apps by User-Agent: put your MB username or email in the brackets
MUSICBRAINZ_USER_AGENT = os.getenv("MUSICBRAINZ_USER_AGENT", "JuggersBot/1.0 ( you@example.com )")

MUSIC_MAX_UPLOAD_BYTES = 20 * 1024 * 1024   # hard cap; also respects the server's own limit
MUSIC_SEARCH_TIMEOUT_MS = 15000            # how long Soulseek search runs
MUSIC_STALL_TIMEOUT = 60                   # secs with no progress before trying the next peer
MUSIC_PICK_COUNT = 10                      # options shown to pick from (cached copy counts as one)
MUSIC_PICK_TIMEOUT = 60                    # secs to wait for a reaction before cancelling
NUMBER_EMOJIS = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"]
MUSIC_QUEUE_SHOW = 15                      # max tracks listed in the !music q embed
MUSIC_SPEED_MIN = 0.1                      # !playback speed lower bound
MUSIC_SPEED_MAX = 10.0                     # !playback speed upper bound
MUSIC_CACHE_FILE = os.getenv("MUSIC_CACHE_FILE", "music_cache.json")  # remembers already-downloaded tracks
MUSIC_TEMP_DIR = os.getenv("MUSIC_TEMP_DIR", "music_tmp")
MUSIC_AUDIO_EXTS = {"flac", "mp3", "m4a", "ogg", "opus", "wav", "aac"}
MUSIC_BAD_WORDS = {"remix", "live", "instrumental", "karaoke", "cover", "nightcore",
                   "sped", "slowed", "acapella", "acoustic", "bootleg", "8d", "reverb"}
os.makedirs(MUSIC_TEMP_DIR, exist_ok=True)

# YouTube specific directories synced to slskd path
YT_MUSIC_DIR = os.path.join(SLSKD_DOWNLOADS_LOCAL, "youtube_music")
YT_VIDEO_DIR = os.path.join(SLSKD_DOWNLOADS_LOCAL, "youtube_video")
os.makedirs(YT_MUSIC_DIR, exist_ok=True)
os.makedirs(YT_VIDEO_DIR, exist_ok=True)
# ---------------------


# The chance that the bot will randomly  (0.15 = 15% chance)
CHAOS_CHANCE = 0.1

# Random chaotic instructions to override the bot's normal behavior
# NOTE: your original list didn't come through in the paste -- put your prompts back in here.
CHAOS_PROMPTS = [
    "Respond with absolute brainrot internet slang. Make almost no sense.",
    "Be extremely cryptic and vague. Give a spooky, unrelated prophecy instead of answering the user."
]

# Async client prevents the discord bot from freezing while thinking
llm_client = AsyncOpenAI(base_url=OLLAMA_URL, api_key="ollama")

# Tracks channels where LLM is active and stores conversation history
llm_active_channels = set()
channel_histories = {}

async def unload_ollama_model(model_name):
    """Forces Ollama to instantly unload the specified model from RAM."""
    try:
        # Ollama API uses /api/generate for this, so we strip /v1 if present
        base_url = OLLAMA_URL.replace("/v1", "")
        url = f"{base_url}/api/generate"
        # keep_alive: 0 immediately frees the model from memory
        payload = {"model": model_name, "keep_alive": 0} 
        async with aiohttp.ClientSession() as session:
            await session.post(url, json=payload)
    except Exception as e:
        print(f"Could not unload model {model_name}: {e}")
# ---------------------

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents, help_command=None)

# ----------------- Core Helper Functions -----------------

def get_asset_file(asset_id, media_type):
    """Fetches the asset from Immich. Videos/GIFs get originals; images get previews."""
    if media_type in ["video", "gif"]:
        url = f"{IMMICH_URL}/api/assets/{asset_id}/original"
        headers = {"x-api-key": IMMICH_API_KEY}
    else:
        url = f"{IMMICH_URL}/api/assets/{asset_id}/thumbnail?size=preview"
        headers = {"x-api-key": IMMICH_API_KEY, "Accept": "image/jpeg"}

    res = requests.get(url, headers=headers)
    res.raise_for_status()

    ext = "jpg"
    if media_type == "video":
        ext = "mp4"
    elif media_type == "gif":
        ext = "gif"

    return discord.File(io.BytesIO(res.content), filename=f"immich_result_{asset_id}.{ext}")


async def execute_search(ctx, endpoint, payload, filter_name, media_type, query):
    """Handles the API request, filters for media type, and sends result."""
    valid_types = {"image": "IMAGE", "gif": "IMAGE", "video": "VIDEO"}
    if media_type.lower() not in valid_types:
        await ctx.send("❌ Invalid media type. Please use `image`, `gif`, or `video`.")
        return

    await ctx.send(f"Searching for **{media_type}** by **{filter_name}** for `{query}`...")

    url = f"{IMMICH_URL}{endpoint}"
    headers = {"x-api-key": IMMICH_API_KEY, "Accept": "application/json"}

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

        assets = data.get("assets", {}).get("items", []) if isinstance(data, dict) else data

        filtered_assets = []
        for asset in assets:
            immich_type = asset.get("type", "")
            original_name = asset.get("originalFileName", "").lower()

            if immich_type == valid_types[media_type.lower()]:
                if media_type.lower() == "gif":
                    if original_name.endswith(".gif"):
                        filtered_assets.append(asset)
                else:
                    filtered_assets.append(asset)

        if not filtered_assets:
            await ctx.send(f"No {media_type} found for `{query}` using the {filter_name} filter.")
            return

        top_asset_id = filtered_assets[0]["id"]

        try:
            file = get_asset_file(top_asset_id, media_type.lower())
            await ctx.send(content=f"Top result for **{query}**:", file=file)
        except requests.exceptions.RequestException:
             await ctx.send("❌ Found an asset, but failed to download it. (It might exceed Discord's 25MB file limit).")

    except Exception as e:
        await ctx.send(f"Failed to communicate with Immich API. Error: {str(e)}")



# ----------------- Music Helper Functions -----------------

class MusicError(Exception):
    """Errors shown to the user as-is."""

_http = None
async def get_http():
    global _http
    if _http is None or _http.closed:
        _http = aiohttp.ClientSession()
    return _http

def _norm(s):
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace("&", " and ")
    return re.sub(r"[\W_]+", " ", s).strip()

def _strip_feat(title):
    return re.sub(r"\s*[(\[]\s*(?:feat|ft|with)\b\.?[^)\]]*[)\]]", "", title, flags=re.I).strip()

def _fmt_time(sec):
    sec = int(sec or 0)
    return f"{sec // 60}:{sec % 60:02d}"

def _bar(pct, width=12):
    filled = int(width * max(0, min(pct, 100)) / 100)
    return "▰" * filled + "▱" * (width - filled)

def _safe_filename(s):
    return re.sub(r'[\\/:*?"<>|]', "", s)[:80] or "track"

def _fmt_speed(s):
    return f"{s:g}x"                                   # 1.5 -> "1.5x", 0.1 -> "0.1x"

def _atempo_chain(speed):
    """ffmpeg's atempo only reliably takes 0.5-2.0 per filter, so chain several for 0.1x-10x."""
    parts = []
    while speed > 2.0:
        parts.append("atempo=2.0")
        speed /= 2.0
    while speed < 0.5:
        parts.append("atempo=0.5")
        speed *= 2.0
    parts.append(f"atempo={speed:.4f}")
    return ",".join(parts)


# ---------- Live status message ----------
class MusicStatus:
    ICONS = {"done": "✅", "run": "⏳", "fail": "❌", "warn": "⚠️"}

    def __init__(self, ctx, query, mode):
        self.ctx, self.query, self.mode = ctx, query, mode
        self.steps = []          # [state, text]
        self.msg = None
        self._last_edit = 0.0
        self.color = discord.Color.blurple()
        self.choices = None      # list of (field title, field text) shown while picking

    def _embed(self):
        desc = "\n".join(f"{self.ICONS[s]} {t}" for s, t in self.steps)[-3000:]
        e = discord.Embed(title=f"🎵 {self.query}"[:256], description=desc, color=self.color)
        where = "Voice channel" if self.mode == "play" else "Voice message"
        e.set_footer(text=f"{where} • requested by {self.ctx.author.display_name}")
        for name, value in (self.choices or []):
            e.add_field(name=name, value=value, inline=False)
        return e

    async def _render(self, force=False):
        now = time.monotonic()
        if not force and now - self._last_edit < 1.5:   # throttle edits to avoid rate limits
            return
        self._last_edit = now
        try:
            if self.msg is None:
                self.msg = await self.ctx.reply(embed=self._embed(), mention_author=False)
            else:
                await self.msg.edit(embed=self._embed())
        except discord.HTTPException:
            pass

    async def step(self, text):
        for s in self.steps:
            if s[0] == "run":
                s[0] = "done"
        self.steps.append(["run", text])
        await self._render(force=True)

    async def update(self, text, force=False):
        if self.steps and self.steps[-1][0] == "run":
            self.steps[-1][1] = text
        else:
            self.steps.append(["run", text])
        await self._render(force)

    async def warn(self, text):
        if self.steps and self.steps[-1][0] == "run":
            self.steps[-1] = ["warn", text]
        else:
            self.steps.append(["warn", text])
        await self._render(force=True)

    async def finish(self, text, ok=True):
        for s in self.steps:
            if s[0] == "run":
                s[0] = "done" if ok else "fail"
        self.steps.append(["done" if ok else "fail", text])
        self.color = discord.Color.green() if ok else discord.Color.red()
        await self._render(force=True)


# ---------- Cache (skip re-downloading tracks you already have) ----------
def _cache_load():
    try:
        with open(MUSIC_CACHE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}

_music_cache = _cache_load()

def _cache_get(key):
    p = _music_cache.get(key)
    return p if p and os.path.isfile(p) else None

def _cache_put(key, path):
    _music_cache[key] = path
    with open(MUSIC_CACHE_FILE, "w") as f:
        json.dump(_music_cache, f, indent=1)

def _cache_clear():
    _music_cache.clear()
    with open(MUSIC_CACHE_FILE, "w") as f:
        json.dump(_music_cache, f, indent=1)

def _purge_cache_dir():
    """Deletes EVERYTHING inside SLSKD_DOWNLOADS_LOCAL (files + folders).
    Returns (files_deleted, bytes_freed, errors)."""
    root = SLSKD_DOWNLOADS_LOCAL
    files_deleted = total_bytes = errors = 0
    if not os.path.isdir(root):
        return 0, 0, 0
        
    for dirpath, _, files in os.walk(root):
        for fn in files:
            p = os.path.join(dirpath, fn)
            try:
                total_bytes += os.path.getsize(p)
                files_deleted += 1
            except OSError:
                pass
                
    for name in os.listdir(root):
        p = os.path.join(root, name)
        try:
            if os.path.isdir(p) and not os.path.islink(p):
                shutil.rmtree(p)
            else:
                os.remove(p)
        except OSError:
            errors += 1
    return files_deleted, total_bytes, errors

# ---------- MusicBrainz ----------
_mb_lock = asyncio.Lock()
_mb_last = 0.0

def _lucene_escape(s):
    return re.sub(r'([+\-&|!(){}\[\]^"~*?:\\/])', r"\\\1", s)

async def mb_search(query):
    """Find the real recording behind a user query. Returns dict or None."""
    global _mb_last
    if " - " in query:
        artist, title = [p.strip() for p in query.split(" - ", 1)]
        lucene = f'recording:"{_lucene_escape(title)}" AND artist:"{_lucene_escape(artist)}"'
    else:
        q = _lucene_escape(query)
        lucene = f"recording:({q})^2 OR artist:({q})"

    session = await get_http()
    async with _mb_lock:                                  # MusicBrainz allows ~1 request/sec
        wait = 1.1 - (time.monotonic() - _mb_last)
        if wait > 0:
            await asyncio.sleep(wait)
        async with session.get(
            "https://musicbrainz.org/ws/2/recording",
            params={"query": lucene, "fmt": "json", "limit": "15"},
            headers={"User-Agent": MUSICBRAINZ_USER_AGENT, "Accept": "application/json"},
            timeout=aiohttp.ClientTimeout(total=15),
        ) as r:
            _mb_last = time.monotonic()
            if r.status != 200:
                return None
            data = await r.json()

    q_tokens = set(_norm(query.replace(" - ", " ")).split())
    asked = set(q_tokens)
    best, best_score, best_cov = None, -1, 0
    for rec in data.get("recordings", []):
        credits = rec.get("artist-credit", [])
        artist = "".join(c.get("name", "") + c.get("joinphrase", "") for c in credits)
        title = rec.get("title", "")
        hay = set(_norm(f"{artist} {title}").split())
        coverage = len(q_tokens & hay) / max(1, len(q_tokens))
        dis = _norm(rec.get("disambiguation", "")).split()
        penalty = 0.4 if any(w in MUSIC_BAD_WORDS and w not in asked for w in dis) else 0
        score = rec.get("score", 0) / 100 + coverage * 1.5 - penalty + (0.05 if rec.get("length") else 0)
        if score > best_score:
            best, best_score, best_cov = rec, score, coverage

    if not best or best_cov < 0.75:
        return None

    credits = best.get("artist-credit", [])
    return {
        "mbid": best["id"],
        "title": best.get("title", ""),
        "artist": "".join(c.get("name", "") + c.get("joinphrase", "") for c in credits),
        "primary_artist": credits[0].get("name", "") if credits else "",
        "length": (best.get("length") or 0) / 1000 or None,     # seconds
        "album": (best.get("releases") or [{}])[0].get("title"),
    }


# ---------- slskd ----------
async def slskd(method, path, **kw):
    session = await get_http()
    async with session.request(
        method, f"{SLSKD_URL}/api/v0{path}",
        headers={"X-API-Key": SLSKD_API_KEY},
        timeout=aiohttp.ClientTimeout(total=30), **kw,
    ) as r:
        if r.status >= 400:
            raise RuntimeError(f"slskd {method} {path} -> {r.status}")
        if r.content_type == "application/json":
            return await r.json()
        return None

async def slskd_search(text, status):
    sid = str(uuid.uuid4())
    await slskd("POST", "/searches", json={
        "id": sid, "searchText": text,
        "searchTimeout": MUSIC_SEARCH_TIMEOUT_MS,
        "responseLimit": 200, "fileLimit": 5000,
    })
    try:
        deadline = time.monotonic() + MUSIC_SEARCH_TIMEOUT_MS / 1000 + 15
        while time.monotonic() < deadline:
            await asyncio.sleep(1.5)
            s = await slskd("GET", f"/searches/{sid}") or {}
            await status.update(
                f"Searching Soulseek for `{text}`… **{s.get('responseCount', 0)}** peers, "
                f"**{s.get('fileCount', 0)}** files")
            if s.get("isComplete"):
                break
        return await slskd("GET", f"/searches/{sid}/responses") or []
    finally:
        try:
            await slskd("DELETE", f"/searches/{sid}")
        except Exception:
            pass

# ---------- Candidate scoring ----------
def _tok_in(tok, hay):
    if tok in hay:
        return True
    if len(tok) >= 4:
        return bool(difflib.get_close_matches(tok, hay, n=1, cutoff=0.8))
    return False

def _coverage(tokens, hay):
    if not tokens:
        return 1.0
    return sum(1 for t in tokens if _tok_in(t, hay)) / len(tokens)

def _allowed_missing(n):
    return 0 if n <= 2 else 1 if n <= 5 else 2

def score_candidates(responses, mb, query, loose=False):
    """Filter out junk and rank the remaining files. Returns best file per peer.
    loose=True relaxes the matching rules (results get flagged in the picker)."""
    title = _strip_feat(mb["title"]) if mb else query
    artist = (mb.get("primary_artist") or "") if mb else ""
    length = mb["length"] if mb else None

    t_tokens = set(_norm(title).split())
    a_tokens = set(_norm(artist).split())
    q_tokens = set(_norm(query.replace(" - ", " ")).split())
    mb_tokens = t_tokens | a_tokens
    bad = {w for w in MUSIC_BAD_WORDS if w not in q_tokens}
    ignore = {"feat", "ft", "original", "official", "audio", "mix", "version"}
    allowed = _allowed_missing(len(t_tokens))

    out = []
    for resp in responses:
        user = resp.get("username")
        free = resp.get("hasFreeUploadSlot", False)
        speed = resp.get("uploadSpeed", 0) or 0
        qlen = resp.get("queueLength", 0) or 0
        for f in resp.get("files", []):
            name = f.get("filename", "")
            ext = (f.get("extension") or name.rsplit(".", 1)[-1]).lower().lstrip(".")
            size = f.get("size", 0) or 0
            if ext not in MUSIC_AUDIO_EXTS or size <= 0 or size > 300 * 1024 * 1024:
                continue

            parts = re.split(r"[\\/]", name)
            base_tokens = set(_norm(parts[-1].rsplit(".", 1)[0]).split())
            path_tokens = set(_norm(" ".join(parts[-3:])).split())

            if loose:
                # Title must be at least half-present, and the whole thing ≥60% present in the path
                if _coverage(t_tokens, path_tokens) < 0.5:
                    continue
                cov = _coverage(q_tokens, path_tokens)
                if mb:
                    cov = max(cov, _coverage(mb_tokens, path_tokens))
                if cov < 0.6:
                    continue
            else:
                hay = base_tokens if mb else path_tokens
                missing = sum(1 for t in t_tokens if not _tok_in(t, hay))
                if missing > allowed:
                    continue
                if a_tokens and _coverage(a_tokens, path_tokens) < 0.5:
                    continue
                if (base_tokens & bad) - t_tokens:      # remix/live/etc not asked for
                    continue

            score = 0.0
            if loose:
                score -= 20
                if (base_tokens & bad) - t_tokens:
                    score -= 25
            elif t_tokens and all(_tok_in(t, base_tokens) for t in t_tokens):
                score += 8                              # clean title match in the filename

            flen = f.get("length")
            if length and flen:
                diff = abs(flen - length)
                if diff > 25 and not loose:             # clearly a different edit / version
                    continue
                score += 30 - min(diff, 25) * 2
            elif length:
                score -= 5

            br = f.get("bitRate") or 0
            # We transcode to Opus anyway, so 320 MP3 is as good as FLAC and downloads faster
            if br >= 320:
                score += 22
            elif ext == "flac":
                score += 20
            elif br >= 256:
                score += 15
            elif br >= 192:
                score += 8
            elif br and br < 160:
                score -= 15

            score += 15 if free else -min(qlen, 50) * 0.3
            score += min(speed / 1_000_000, 10) * 1.5
            junk = {t for t in base_tokens - t_tokens - a_tokens - ignore if not t.isdigit()}
            score -= len(junk) * 1.5

            desc = "FLAC" if ext == "flac" else f"{ext.upper()} {br}kbps" if br else ext.upper()
            out.append({"username": user, "filename": name, "size": size,
                        "score": score, "desc": desc, "length": flen, "free": free,
                        "loose": loose})

    out.sort(key=lambda c: c["score"], reverse=True)
    seen, unique = set(), []
    for c in out:                                  # one file per peer → better fallbacks
        if c["username"] not in seen:
            seen.add(c["username"])
            unique.append(c)
    return unique

def gather_candidates(responses, mb, query, want):
    """Strict pass first; if that gives fewer than `want`, top up with loose matches."""
    cands = score_candidates(responses, mb, query)
    if len(cands) >= want:
        return cands
    extra = score_candidates(responses, mb, query, loose=True)
    if mb:                                         # MusicBrainz may have picked the wrong recording
        extra += score_candidates(responses, None, query, loose=True)
    extra.sort(key=lambda c: c["score"], reverse=True)
    seen = {c["username"] for c in cands}
    for c in extra:
        if len(cands) >= want:
            break
        if c["username"] not in seen:
            seen.add(c["username"])
            cands.append(c)
    return cands

async def _find_transfer(user, filename):
    try:
        data = await slskd("GET", f"/transfers/downloads/{quote(user, safe='')}")
    except Exception:
        return None
    for d in (data or {}).get("directories", []):
        for f in d.get("files", []):
            if f.get("filename") == filename:
                return f
    return None

async def _cancel_transfer(user, tid):
    try:
        await slskd("DELETE", f"/transfers/downloads/{quote(user, safe='')}/{tid}",
                    params={"remove": "true"})
    except Exception:
        pass

async def slskd_download(cand, status):
    user, filename = cand["username"], cand["filename"]
    old = await _find_transfer(user, filename)
    if old:
        await _cancel_transfer(user, old.get("id"))

    await slskd("POST", f"/transfers/downloads/{quote(user, safe='')}",
                json=[{"filename": filename, "size": cand["size"]}])

    start = last_change = time.monotonic()
    last_bytes = -1
    
    # Prime psutil CPU calculation
    psutil.cpu_percent(interval=None)
    get_sys_telemetry()

    while True:
        await asyncio.sleep(2)
        t = await _find_transfer(user, filename)
        if t is None:
            if time.monotonic() - start > 20:
                raise RuntimeError("transfer disappeared")
            continue

        state = t.get("state", "")
        pct = t.get("percentComplete", 0) or 0
        if "Succeeded" in state:
            await status.update(f"Downloaded from **{user}** ({cand['desc']}) {_bar(100)} 100%", force=True)
            return
        if state.startswith("Completed"):
            raise RuntimeError(state.replace("Completed, ", "").lower())

        transferred = t.get("bytesTransferred", 0)
        if transferred != last_bytes:
            last_bytes, last_change = transferred, time.monotonic()
        if time.monotonic() - last_change > MUSIC_STALL_TIMEOUT:
            await _cancel_transfer(user, t.get("id"))
            raise RuntimeError("peer stalled / queued too long")

        nice_state = "queued by peer" if "Queued" in state else "downloading"
        stats = get_sys_stats_string()
        await status.update(
            f"Downloading from **{user}** ({cand['desc']}) {_bar(pct)} {pct:.0f}% — {nice_state}\n{stats}")

def _local_path_for(remote):
    """slskd saves to <downloads>/<remote parent folder>/<file>; fall back to a scan."""
    parts = re.split(r"[\\/]", remote)
    base = parts[-1]
    if len(parts) >= 2:
        p = os.path.join(SLSKD_DOWNLOADS_LOCAL, parts[-2], base)
        if os.path.isfile(p):
            return p
    best = None
    for root, _, files in os.walk(SLSKD_DOWNLOADS_LOCAL):
        if base in files:
            p = os.path.join(root, base)
            if best is None or os.path.getmtime(p) > os.path.getmtime(best):
                best = p
    return best

async def _wait_for_local_file(remote):
    loop = asyncio.get_running_loop()
    for _ in range(10):                                  # slskd may still be moving the file
        path = await loop.run_in_executor(None, _local_path_for, remote)
        if path:
            return path
        await asyncio.sleep(1)
    return None


# ---------- ffmpeg ----------
async def _run(*args):
    p = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    out, err = await p.communicate()
    if p.returncode != 0:
        raise RuntimeError(err.decode(errors="ignore")[-300:])
    return out

async def probe_duration(path):
    out = await _run("ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=nw=1:nk=1", path)
    return float(out.strip() or 0)

async def probe_info(path):
    """Duration / codec / bitrate / sample rate for np + picker."""
    out = await _run("ffprobe", "-v", "error", "-select_streams", "a:0",
                     "-show_entries", "format=duration,bit_rate:stream=codec_name,sample_rate",
                     "-of", "json", path)
    data = json.loads(out or b"{}")
    fmt = data.get("format") or {}
    st = (data.get("streams") or [{}])[0]
    br = int(float(fmt.get("bit_rate") or 0))
    return {"duration": float(fmt.get("duration") or 0),
            "codec": st.get("codec_name") or "",
            "kbps": round(br / 1000) if br else 0,
            "sample_rate": int(st.get("sample_rate") or 0)}

def _fmt_desc(info):
    codec = (info.get("codec") or "").lower()
    if not codec:
        return "Unknown"
    if codec.startswith("pcm"):
        return "WAV"
    if codec == "flac" or not info.get("kbps"):
        return codec.upper()
    return f"{codec.upper()} {info['kbps']}kbps"

async def transcode_for_discord(src, limit_bytes, duration):
    """Convert to OGG/Opus (required for voice messages), sized under limit_bytes."""
    kbps = min(128, int(limit_bytes * 8 * 0.92 / max(duration, 1) / 1000))
    if kbps < 24:
        raise MusicError(f"Track is too long ({_fmt_time(duration)}) to fit under the upload limit.")
    out = os.path.join(MUSIC_TEMP_DIR, f"{uuid.uuid4().hex}.ogg")
    for _ in range(3):
        await _run("ffmpeg", "-y", "-v", "error", "-i", src, "-vn", "-map_metadata", "-1",
                   "-ac", "2", "-ar", "48000", "-c:a", "libopus", "-b:a", f"{kbps}k", out)
        if os.path.getsize(out) <= limit_bytes:
            return out, kbps
        kbps = int(kbps * 0.8)
    os.remove(out)
    raise MusicError("Couldn't get the file under the upload limit.")

async def make_waveform(path):
    """256-byte base64 waveform Discord shows on voice messages."""
    raw = await _run("ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", "4000",
                     "-f", "s16le", "-")
    samples = array.array("h")
    samples.frombytes(raw[: len(raw) // 2 * 2])
    if not samples:
        return base64.b64encode(bytes(256)).decode()
    step = max(1, len(samples) // 256)
    vals = []
    for i in range(256):
        chunk = samples[i * step:(i + 1) * step]
        peak = max((abs(x) for x in chunk), default=0)
        vals.append(min(255, peak * 255 // 32768))
    return base64.b64encode(bytes(vals)).decode()

async def send_voice_message(channel_id, path, duration, waveform):
    """discord.py can't send voice messages natively, so call the API directly."""
    session = await get_http()
    with open(path, "rb") as fh:
        data = fh.read()
    form = aiohttp.FormData()
    form.add_field("payload_json", json.dumps({
        "flags": 1 << 13,                               # IS_VOICE_MESSAGE
        "attachments": [{"id": 0, "filename": "voice-message.ogg",
                         "duration_secs": round(duration, 2), "waveform": waveform}],
    }), content_type="application/json")
    form.add_field("files[0]", data, filename="voice-message.ogg", content_type="audio/ogg")
    async with session.post(
        f"https://discord.com/api/v10/channels/{channel_id}/messages",
        data=form, headers={"Authorization": f"Bot {DISCORD_BOT_TOKEN}"},
    ) as r:
        if r.status >= 300:
            raise RuntimeError(f"Discord {r.status}: {(await r.text())[:200]}")

# ---------- Reaction picker ----------
def _emoji_key(e):
    return str(e).replace("\ufe0f", "")       # keycap emojis sometimes arrive without FE0F

def _chunk_lines(lines, limit=1024):
    chunks, cur = [], ""
    for ln in lines:
        if cur and len(cur) + len(ln) + 1 > limit:
            chunks.append(cur)
            cur = ln
        else:
            cur = f"{cur}\n{ln}" if cur else ln
    if cur:
        chunks.append(cur)
    return chunks

async def pick_candidate(status, options):
    """Show numbered options, wait for any user's reaction. Returns (candidate, picker_name) or None."""
    lines = []
    for i, c in enumerate(options):
        if c.get("cached"):
            shown = os.path.basename(c["path"]).replace("`", "'")
        else:
            parts = re.split(r"[\\/]", c["filename"])
            shown = "/".join(parts[-2:]).replace("`", "'")
        if len(shown) > 60:
            shown = "…" + shown[-59:]
        info = [c["desc"]]
        if c.get("length"):
            info.append(_fmt_time(c["length"]))
        if c.get("size"):  # Only show MB if we actually know the file size
            info.append(f"{c['size'] / 1048576:.1f} MB")
        if c.get("cached"):
            info.append("📦 cached")
        else:
            info.append(("🟢 " if c["free"] else "🕒 ") + discord.utils.escape_markdown(c["username"][:30]))
            if c.get("loose"):
                info.append("⚠️ loose")
        lines.append(f"{NUMBER_EMOJIS[i]} `{shown}`\n{' · '.join(info)}")

    emojis = NUMBER_EMOJIS[:len(options)]
    title = "Pick a source (📦 cached · 🟢 free slot · 🕒 queued · ⚠️ loose match)"
    status.choices = [(title if n == 0 else "\u200b", chunk)
                      for n, chunk in enumerate(_chunk_lines(lines))]
    await status.step(f"React with {emojis[0]}–{emojis[-1]} to choose ({MUSIC_PICK_TIMEOUT}s)…")

    msg = status.msg
    if msg is None:                                # status message failed to send
        status.choices = None
        return options[0], "Auto"

    valid = [_emoji_key(e) for e in emojis]
    def check(p):
        return p.message_id == msg.id and p.user_id != bot.user.id and _emoji_key(p.emoji) in valid

    # Start listening before adding reactions so fast clicks aren't missed
    waiter = asyncio.create_task(
        bot.wait_for("raw_reaction_add", check=check, timeout=MUSIC_PICK_TIMEOUT))
    for e in emojis:
        try:
            await msg.add_reaction(e)
        except discord.HTTPException:
            pass

    try:
        payload = await waiter
    except asyncio.TimeoutError:
        payload = None
    finally:
        status.choices = None
        try:
            await msg.clear_reactions()            # needs Manage Messages
        except discord.HTTPException:
            for e in emojis:                       # fallback: just remove the bot's own
                try:
                    await msg.remove_reaction(e, bot.user)
                except discord.HTTPException:
                    pass

    if payload is None:
        return None
    who = payload.member.display_name if payload.member else "Someone"
    return options[valid.index(_emoji_key(payload.emoji))], who

# ---------- Pipeline ----------
_music_sem = asyncio.Semaphore(2)        # max simultaneous track requests

async def _cached_candidate(path):
    cand = {"cached": True, "path": path, "username": "cache", "filename": path,
            "size": os.path.getsize(path), "score": 999, "free": True, "loose": False,
            "desc": os.path.splitext(path)[1].lstrip(".").upper() or "AUDIO", "length": None}
    try:
        info = await probe_info(path)
        cand["desc"] = _fmt_desc(info)
        cand["length"] = info["duration"] or None
    except Exception:
        pass
    return cand

async def resolve_track(query, status):
    """MusicBrainz → Soulseek search (+ cached copy) → pick → download. Returns (local_path, label)."""

    # Force raw search variables since we skipped MusicBrainz
    mb = None
    label = query
    cache_key = "q:" + _norm(query)
    search_text = _norm(query.replace(" - ", " "))
    
    #await status.step("Looking up track on MusicBrainz…")
    #try:
        #mb = await mb_search(query)
    #except Exception:
        #mb = None

    #if mb:
        #label = f"{mb['artist']} - {mb['title']}"
        #extra = f" ({_fmt_time(mb['length'])})" if mb["length"] else ""
        #album = f" from *{mb['album']}*" if mb.get("album") else ""
        #await status.update(f"MusicBrainz match: **{label}**{extra}{album}", force=True)
        #cache_key = mb["mbid"]
        #search_text = _norm(f"{mb['primary_artist']} {_strip_feat(mb['title'])}")
    #else:
        #await status.warn("No confident MusicBrainz match — using raw search (less accurate)")
        #label = query
        #cache_key = "q:" + _norm(query)
        #search_text = _norm(query.replace(" - ", " "))
    

    cached = _cache_get(cache_key)
    cached_cand = None
    if cached:
        cached_cand = await _cached_candidate(cached)
        await status.step("Found a cached copy — it'll be listed first")

    await status.step(f"forget MUSICBRAINZ, entering your search in soulseek for `{search_text}`… hold up a sec fella")
    responses = []
    try:
        responses = await slskd_search(search_text, status)
    except Exception as e:
        if not cached_cand:
            raise
        await status.warn(f"Soulseek search failed ({e}) — only the cached copy is available")

    loop = asyncio.get_running_loop()
    cands = await loop.run_in_executor(None, gather_candidates, responses, mb, query, MUSIC_PICK_COUNT)
    if cached_cand:                                # don't list the cached file twice
        cb = os.path.basename(cached)
        cands = [c for c in cands if re.split(r"[\\/]", c["filename"])[-1] != cb]
    want = MUSIC_PICK_COUNT - (1 if cached_cand else 0)
    cands = cands[:want]

    if not cands and not cached_cand:
        raise MusicError(f"No usable files found on Soulseek ({len(responses)} peers checked). "
                         "Try `!music Artist - Title`.")
    n_loose = sum(1 for c in cands if c.get("loose"))
    loose_txt = f" ({n_loose} loose ⚠️)" if n_loose else ""
    await status.update(f"Found **{len(cands)}** sources out of {len(responses)} peers{loose_txt}"
                        + (" + cached copy" if cached_cand else ""), force=True)

    options = ([cached_cand] if cached_cand else []) + cands
    while options:
        picked = await pick_candidate(status, options)
        if picked is None:
            raise MusicError(f"Nobody picked a source within {MUSIC_PICK_TIMEOUT}s — request cancelled.")
        c, who = picked
        if c.get("cached"):
            await status.step(f"**{discord.utils.escape_markdown(who)}** picked the **cached copy** ({c['desc']})")
            return c["path"], label
        await status.step(f"**{discord.utils.escape_markdown(who)}** picked "
                          f"**{discord.utils.escape_markdown(c['username'])}** ({c['desc']}) — requesting…")
        try:
            await slskd_download(c, status)
            path = await _wait_for_local_file(c["filename"])
            if not path:
                raise MusicError("Downloaded, but the bot can't see the file. "
                                 "Check SLSKD_DOWNLOADS_LOCAL matches slskd's downloads folder.")
            _cache_put(cache_key, path)
            return path, label
        except MusicError:
            raise
        except Exception as e:
            options.remove(c)
            more = " — pick another" if options else ""
            await status.warn(f"That source failed ({e}){more}")
    raise MusicError("Every source failed. Try again later or be more specific.")

async def send_track_as_voice(ctx, path, label, status):
    limit = min(MUSIC_MAX_UPLOAD_BYTES, ctx.guild.filesize_limit if ctx.guild else 20 * 1024 * 1024)
    await status.step("Converting to voice message…")
    duration = await probe_duration(path)
    ogg, kbps = await transcode_for_discord(path, limit, duration)
    try:
        size_mb = os.path.getsize(ogg) / 1048576
        await status.update(f"Converted — Opus {kbps} kbps, {size_mb:.1f} MB "
                            f"(limit {limit / 1048576:.0f} MB)", force=True)
        waveform = await make_waveform(ogg)
        await status.step("Uploading to Discord…")
        try:
            await send_voice_message(ctx.channel.id, ogg, duration, waveform)
        except Exception:
            await status.warn("Voice message rejected — sending as a normal audio file")
            await ctx.send(file=discord.File(ogg, filename=f"{_safe_filename(label)}.ogg"))
        await status.finish(f"Sent **{label}** ({_fmt_time(duration)})")
    finally:
        if os.path.exists(ogg):
            os.remove(ogg)

# ---------- Player state & queue ----------
class GuildPlayer:
    def __init__(self):
        self.queue = []          # tracks waiting to play
        self.current = None      # track dict that's playing right now
        self.speed = 1.0         # live playback speed (!playback speed), applies to every track
        self.lock = asyncio.Lock()

music_players = {}               # guild_id → GuildPlayer

def get_player(guild_id):
    p = music_players.get(guild_id)
    if p is None:
        p = music_players[guild_id] = GuildPlayer()
    return p

def _track_position(track):
    """Where we are in the SOURCE file (seconds), accounting for playback speed."""
    if track.get("seg_start") is None:
        return 0.0
    pos = track["pos_base"] + (time.monotonic() - track["seg_start"]) * track["speed"]
    dur = track["duration"] or 0
    return max(0.0, min(pos, dur) if dur else pos)

def _playing_text(track, vc):
    txt = f"🔊 Playing **{track['label']}** ({_fmt_time(track['duration'])}) in **{vc.channel.name}**"
    if track["speed"] != 1:
        txt += f" · ⏩ {_fmt_speed(track['speed'])}"
    if track["loops_left"]:
        txt += f" · 🔁 {track['loops_left']} loop{'s' if track['loops_left'] != 1 else ''} left"
    return txt

def _make_source(path, start, speed):
    """FFmpeg source starting at `start` seconds, with pitch-preserving tempo change if needed."""
    before = f"-ss {start:.3f}" if start > 0 else None
    opts = f"-vn -af {_atempo_chain(speed)}" if abs(speed - 1.0) > 1e-9 else None
    return discord.FFmpegOpusAudio(path, before_options=before, options=opts)

async def _start_track(vc, player, track, start=0.0):
    player.current = track
    track["seg"] += 1                                # each ffmpeg run gets its own id
    seg = track["seg"]
    track["pos_base"] = start
    track["seg_start"] = time.monotonic()
    track["speed"] = player.speed

    def _after(err):
        asyncio.run_coroutine_threadsafe(_on_track_end(vc, player, track, err, seg), bot.loop)

    try:
        vc.play(_make_source(track["path"], start, track["speed"]), after=_after)
    except Exception:
        player.current = None
        raise
    await track["status"].update(_playing_text(track, vc), force=True)

async def _restart_current(vc, player, start):
    """Re-launches the current track at `start` (used for live speed changes)."""
    track = player.current
    track["seg"] += 1                                # old source's end-callback becomes a no-op
    vc.stop()
    await _start_track(vc, player, track, start=start)

async def _on_track_end(vc, player, track, err, seg):
    if player.current is not track or track["seg"] != seg:   # stopped / replaced / restarted
        return

    # Loop: replay the same track (skip sets loops_left = 0 and skipped = True, so it never loops)
    if (not err and not track["skipped"] and track["loops_left"] > 0
            and vc.is_connected() and os.path.isfile(track["path"])):
        track["loops_left"] -= 1
        try:
            await _start_track(vc, player, track)
            return
        except Exception as e:
            err = e

    player.current = None
    if err:
        await track["status"].finish(f"Playback error: `{err}`", ok=False)
    elif track["skipped"]:
        await track["status"].finish(f"Skipped **{track['label']}**")
    else:
        await track["status"].finish(f"Finished playing **{track['label']}**")

    async with player.lock:
        while player.queue and vc.is_connected():
            nxt = player.queue.pop(0)
            if not os.path.isfile(nxt["path"]):
                await nxt["status"].finish("File is gone from the cache — skipped", ok=False)
                continue
            try:
                await _start_track(vc, player, nxt)
                return
            except Exception as e:
                await nxt["status"].finish(f"Couldn't play: `{e}`", ok=False)
        if player.current is None and vc.is_connected():   # nothing left (and nothing new started)
            await vc.disconnect()

async def play_in_vc(ctx, path, label, status, thumbnail=None):
    if not (ctx.author.voice and ctx.author.voice.channel):
        raise MusicError("You left the voice channel before the track was ready.")
    channel = ctx.author.voice.channel
    player = get_player(ctx.guild.id)

    info = await probe_info(path)
    track = {"path": path, "label": label, "duration": info["duration"],
             "size": os.path.getsize(path), "format": _fmt_desc(info),
             "sample_rate": info["sample_rate"], "requester": ctx.author.display_name,
             "status": status,
             "seg": 0, "pos_base": 0.0, "seg_start": None, "speed": 1.0, 
             "loops_left": 0, "skipped": False, "thumbnail": thumbnail}   # !music loop / !music skip

    async with player.lock:
        vc = ctx.voice_client
        busy = vc and vc.is_connected() and (vc.is_playing() or vc.is_paused() or player.current)
        if busy:
            if vc.channel != channel:
                raise MusicError(f"I'm already playing in **{vc.channel.name}** — "
                                 "join that channel to add tracks to the queue.")
            player.queue.append(track)
            await status.step(f"Queued at position **#{len(player.queue)}** — "
                              f"**{label}** ({_fmt_time(track['duration'])})")
            return

        await status.step(f"Joining **{channel.name}**…")
        if vc and vc.channel != channel:
            await vc.move_to(channel)
        elif not vc:
            vc = await channel.connect(self_deaf=True)
        player.speed = 1.0                         # fresh session → back to normal speed
        await _start_track(vc, player, track)

# ---------- Embeds ----------
def build_np_embed(player):
    t = player.current
    dur = t["duration"] or 0
    elapsed = _track_position(t)
    pct = elapsed / dur * 100 if dur else 0

    e = discord.Embed(
        title="🎵 Now Playing",
        description=(f"**{discord.utils.escape_markdown(t['label'])}**\n\n"
                     f"▶️ `{_fmt_time(elapsed)}` {_bar(pct, 16)} `{_fmt_time(dur)}` ({pct:.0f}%)"),
        color=discord.Color.blurple())
    fmt = t["format"]
    if t.get("sample_rate"):
        fmt += f" · {t['sample_rate'] / 1000:.1f} kHz"
    e.add_field(name="Length", value=_fmt_time(dur), inline=True)
    e.add_field(name="Size", value=f"{t['size'] / 1048576:.1f} MB", inline=True)
    e.add_field(name="Format", value=fmt, inline=True)
    if t["speed"] != 1:
        e.add_field(name="Speed", value=_fmt_speed(t["speed"]), inline=True)
    if t["loops_left"]:
        e.add_field(name="Loops left", value=str(t["loops_left"]), inline=True)
    if player.queue:
        nxt = discord.utils.escape_markdown(player.queue[0]["label"])
        e.add_field(name="Up next", value=f"{nxt}\n({len(player.queue)} in queue)", inline=False)
    if t.get("thumbnail"):
        e.set_thumbnail(url=t["thumbnail"])
    e.set_footer(text=f"Voice channel • requested by {t['requester']}")
    return e

def build_queue_embed(player):
    lines = []
    if player.current:
        t = player.current
        lines.append(f"🔊 **Now playing:** {discord.utils.escape_markdown(t['label'])} "
                     f"({_fmt_time(t['duration'])})")
    if player.queue:
        if lines:
            lines.append("")
        for i, t in enumerate(player.queue[:MUSIC_QUEUE_SHOW], 1):
            lines.append(f"`{i}.` {discord.utils.escape_markdown(t['label'])} "
                         f"· {_fmt_time(t['duration'])} · *{discord.utils.escape_markdown(t['requester'])}*")
        hidden = len(player.queue) - MUSIC_QUEUE_SHOW
        if hidden > 0:
            lines.append(f"…and **{hidden}** more")
    total = sum(t["duration"] or 0 for t in player.queue)
    e = discord.Embed(title="🎶 Music Queue", description="\n".join(lines)[:4000],
                      color=discord.Color.blurple())
    e.set_footer(text=f"Voice channel • {len(player.queue)} queued • {_fmt_time(total)} total")
    return e

async def run_music(ctx, query, mode):
    status = MusicStatus(ctx, query, mode)
    if mode == "play" and not (ctx.author.voice and ctx.author.voice.channel):
        await status.finish("Join a voice channel first, then use `!music play`.", ok=False)
        return
    if _music_sem.locked():
        await status.step("Waiting for another request to finish…")
    async with _music_sem:
        try:
            path, label = await resolve_track(query, status)
            if mode == "play":
                await play_in_vc(ctx, path, label, status)
            else:
                await send_track_as_voice(ctx, path, label, status)
        except MusicError as e:
            await status.finish(str(e), ok=False)
        except Exception as e:
            await status.finish(f"Something broke: `{e}`", ok=False)

def _sanitize_yt_query(query):
    """Extracts the base URL or video ID to prevent playlist spam, or formats for search."""
    if "youtube.com" in query or "youtu.be" in query:
        # Regex to safely extract just the video ID from any messy youtube URL
        match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11})', query)
        if match:
            return f"https://www.youtube.com/watch?v={match.group(1)}"
        return query.split('&')[0]
    return f"ytsearch10:{query}"

async def resolve_yt_track(query, status):
    """Searches YouTube, lets user pick from 10 results, checks cache, then downloads."""
    loop = asyncio.get_running_loop()
    search_query = _sanitize_yt_query(query)
    is_url = search_query.startswith('http')

    await status.step(f"Searching YouTube for `{'URL' if is_url else query}`...")
    
    def _search():
        ydl_opts = {'extract_flat': True, 'quiet': True, 'no_warnings': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(search_query, download=False)
            if 'entries' in info:
                info['entries'] = [e for e in info['entries'] if e] 
            return info
            
    try:
        info = await loop.run_in_executor(None, _search)
    except Exception as e:
        raise MusicError(f"YouTube search failed: {e}")
        
    entries = info.get('entries', []) if 'entries' in info else [info]
    if not entries:
        raise MusicError("No results found on YouTube.")

    options = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
            
        yt_url = entry.get("url") or (f"https://www.youtube.com/watch?v={entry.get('id')}" if entry.get("id") else None)
        if not yt_url: continue
            
        thumb = entry.get("thumbnail") or (entry.get("thumbnails")[-1].get("url") if entry.get("thumbnails") else None)
        uploader = entry.get("uploader") or entry.get("channel") or "YouTube"

        thumb_link = f"[🖼️ View Thumb]({thumb})" if thumb else "🎵"

        options.append({
            "yt_url": yt_url,
            "filename": entry.get("title", "Unknown Title"), 
            "desc": f"{thumb_link} • 👤 {uploader}",
            "length": entry.get("duration"),
            "size": 0, 
            "username": uploader, 
            "free": True, 
            "loose": False,
            "cached": False,
            "thumbnail": thumb
        })
        
    if not options:
        raise MusicError("No valid YouTube videos found.")

    if len(options) == 1 and is_url:
        c, who = options[0], "Auto"
    else:
        picked = await pick_candidate(status, options)
        if picked is None:
            raise MusicError(f"Nobody picked a source within {MUSIC_PICK_TIMEOUT}s — cancelled.")
        c, who = picked
        
    # --- Cache Check ---
    cache_key = f"yt:aud:{c['yt_url']}"
    cached_path = _cache_get(cache_key)
    
    if cached_path:
        await status.step(f"**{discord.utils.escape_markdown(who)}** picked **{discord.utils.escape_markdown(c['filename'])}** (Cached!)")
        return cached_path, c["filename"], c["thumbnail"]

    await status.step(f"**{discord.utils.escape_markdown(who)}** picked **{discord.utils.escape_markdown(c['filename'])}** — downloading...")

    # Prime psutil and telemetry
    psutil.cpu_percent(interval=None)
    get_sys_telemetry()
    
    last_dl_update = 0
    def yt_hook(d):
        nonlocal last_dl_update
        if d['status'] == 'downloading':
            now = time.monotonic()
            if now - last_dl_update > 1.2:
                last_dl_update = now
                try:
                    clean_pct = re.sub(r'\x1b\[[0-9;]*m', '', d.get('_percent_str', '0%')).replace('%', '').strip()
                    pct = float(clean_pct) if clean_pct != 'Unknown' else 0.0
                    spd = re.sub(r'\x1b\[[0-9;]*m', '', d.get('_speed_str', '0B/s')).strip()
                    eta = re.sub(r'\x1b\[[0-9;]*m', '', d.get('_eta_str', 'Unknown')).strip()
                    sz = re.sub(r'\x1b\[[0-9;]*m', '', d.get('_total_bytes_str', d.get('_total_bytes_estimate_str', '0B'))).strip()

                    stats = get_sys_stats_string()
                    msg = f"🔽 Downloading Audio... {_bar(pct)} {pct:.1f}% | ⚡ {spd} | ⏳ ETA: {eta} | 📦 {sz}\n{stats}"
                    asyncio.run_coroutine_threadsafe(status.update(msg), bot.loop)
                except Exception:
                    pass
        elif d['status'] == 'finished':
            msg = f"✅ Download Complete {_bar(100)} 100% | Finalizing..."
            asyncio.run_coroutine_threadsafe(status.update(msg, force=True), bot.loop)

    def _download():
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(YT_MUSIC_DIR, 'yt_%(id)s.%(ext)s'),
            'noplaylist': True,
            'quiet': True,
            'no_warnings': True,
            'progress_hooks': [yt_hook]
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            dl_info = ydl.extract_info(c["yt_url"], download=True)
            return ydl.prepare_filename(dl_info)
            
    try:
        path = await loop.run_in_executor(None, _download)
        _cache_put(cache_key, path)
        await status.update(f"✅ Ready: **{c['filename']}**", force=True)
        return path, c["filename"], c["thumbnail"]
    except Exception as e:
        raise MusicError(f"YouTube download failed: {e}")

async def run_yt_music(ctx, query, mode):
    """Pipeline for YouTube tracks, mirroring run_music."""
    status = MusicStatus(ctx, query, mode)
    if mode == "play" and not (ctx.author.voice and ctx.author.voice.channel):
        await status.finish("Join a voice channel first, then use `!ytmusic play`.", ok=False)
        return
    
    if _music_sem.locked():
        await status.step("Waiting for another request to finish…")
        
    async with _music_sem:
        try:
            path, label, thumbnail = await resolve_yt_track(query, status)
            
            if mode == "play":
                await play_in_vc(ctx, path, label, status, thumbnail=thumbnail)
            else:
                await send_track_as_voice(ctx, path, label, status)
        except MusicError as e:
            await status.finish(str(e), ok=False)
        except Exception as e:
            await status.finish(f"Something broke: `{e}`", ok=False)

async def run_yt_video_download(ctx, query, status):
    loop = asyncio.get_running_loop()
    search_query = _sanitize_yt_query(query)
    is_url = search_query.startswith('http')

    # --- 1. Search YouTube ---
    await status.step(f"Searching YouTube for `{'URL' if is_url else query}`...")
    
    def _search():
        ydl_opts = {'extract_flat': True, 'quiet': True, 'no_warnings': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(search_query, download=False)
            if 'entries' in info:
                info['entries'] = [e for e in info['entries'] if e] 
            return info
            
    try:
        info = await loop.run_in_executor(None, _search)
    except Exception as e:
        raise MusicError(f"YouTube search failed: {e}")
        
    entries = info.get('entries', []) if 'entries' in info else [info]
    if not entries:
        raise MusicError("No results found on YouTube.")

    options = []
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("title"):
            continue
            
        yt_url = entry.get("url") or (f"https://www.youtube.com/watch?v={entry.get('id')}" if entry.get("id") else None)
        if not yt_url: continue
            
        thumb = entry.get("thumbnail") or (entry.get("thumbnails")[-1].get("url") if entry.get("thumbnails") else None)
        uploader = entry.get("uploader") or entry.get("channel") or "YouTube"

        thumb_link = f"[🖼️ View Thumb]({thumb})" if thumb else "🎬"

        options.append({
            "yt_url": yt_url,
            "filename": entry.get("title", "Unknown Title"), 
            "desc": f"{thumb_link} • 👤 {uploader}",
            "length": entry.get("duration"),
            "size": 0, 
            "username": uploader, 
            "free": True, 
            "loose": False,
            "cached": False,
            "thumbnail": thumb
        })
        
    if not options:
        raise MusicError("No valid YouTube videos found.")

    if len(options) == 1 and is_url:
        c, who = options[0], "Auto"
    else:
        picked = await pick_candidate(status, options)
        if picked is None:
            raise MusicError(f"Nobody picked a source within {MUSIC_PICK_TIMEOUT}s — cancelled.")
        c, who = picked
        
    # --- 2. Check Cache ---
    cache_key = f"yt:vid:{c['yt_url']}"
    cached_path = _cache_get(cache_key)
    
    if cached_path:
        status.choices = None 
        if status.msg and c["thumbnail"]:
            e = status.msg.embeds[0]
            e.set_thumbnail(url=c["thumbnail"])
            await status.msg.edit(embed=e)
            
        await status.step(f"**{discord.utils.escape_markdown(who)}** picked **{discord.utils.escape_markdown(c['filename'])}** (Cached!)")
        
        safe_name = _safe_filename(c['filename']) + ".mp4"
        file = discord.File(cached_path, filename=safe_name)
        await ctx.send(content=f"🎥 **{c['filename']}**", file=file)
        await status.finish("Cached video sent successfully!")
        return

    # --- 3. Download with Live Progress Bar & Telemetry ---
    status.choices = None 
    if status.msg and c["thumbnail"]:
        e = status.msg.embeds[0]
        e.set_thumbnail(url=c["thumbnail"])
        await status.msg.edit(embed=e)

    await status.step(f"**{discord.utils.escape_markdown(who)}** picked **{discord.utils.escape_markdown(c['filename'])}** — initializing download...")

    # Prime psutil and telemetry
    psutil.cpu_percent(interval=None)
    get_sys_telemetry()

    last_dl_update = 0
    def yt_hook(d):
        nonlocal last_dl_update
        if d['status'] == 'downloading':
            now = time.monotonic()
            if now - last_dl_update > 1.2:
                last_dl_update = now
                try:
                    clean_pct = re.sub(r'\x1b\[[0-9;]*m', '', d.get('_percent_str', '0%')).replace('%', '').strip()
                    pct = float(clean_pct) if clean_pct != 'Unknown' else 0.0
                    spd = re.sub(r'\x1b\[[0-9;]*m', '', d.get('_speed_str', '0B/s')).strip()
                    eta = re.sub(r'\x1b\[[0-9;]*m', '', d.get('_eta_str', 'Unknown')).strip()
                    sz = re.sub(r'\x1b\[[0-9;]*m', '', d.get('_total_bytes_str', d.get('_total_bytes_estimate_str', '0B'))).strip()
                    
                    stats = get_sys_stats_string()
                    msg = f"🔽 Downloading Video... {_bar(pct)} {pct:.1f}% | ⚡ {spd} | ⏳ ETA: {eta} | 📦 {sz}\n{stats}"
                    asyncio.run_coroutine_threadsafe(status.update(msg), bot.loop)
                except Exception:
                    pass
        elif d['status'] == 'finished':
            msg = f"✅ Download Complete {_bar(100)} 100% | Finalizing..."
            asyncio.run_coroutine_threadsafe(status.update(msg, force=True), bot.loop)

    def _download():
        ydl_opts = {
            'format': 'bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': os.path.join(YT_VIDEO_DIR, 'ytvid_%(id)s.%(ext)s'),
            'noplaylist': True,
            'quiet': True,
            'no_warnings': True,
            'progress_hooks': [yt_hook]
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            dl_info = ydl.extract_info(c["yt_url"], download=True)
            return ydl.prepare_filename(dl_info), dl_info.get('duration', 0)
            
    orig_path, duration = await loop.run_in_executor(None, _download)
    final_path = orig_path
    
    compressed = False

    try:
        if not duration:
            duration = await probe_duration(orig_path)
            
        # --- 4. Check Size & Compress if Needed ---
        limit_bytes = 20 * 1024 * 1024
        orig_size = os.path.getsize(orig_path)
        
        if orig_size > limit_bytes:
            # Target 19.0MB strictly to leave room for the MP4 container overhead
            target_bytes = 19.0 * 1024 * 1024 
            total_kbps = int((target_bytes * 8) / max(duration, 1) / 1000)
            
            # Scale audio dynamically but cap at 128k
            audio_kbps = min(128, max(32, int(total_kbps * 0.15))) 
            video_kbps = total_kbps - audio_kbps
            
            if video_kbps < 50:
                raise MusicError(f"Video is too long ({_fmt_time(duration)}) to compress under the 20MB limit.")
                
            await status.step(f"Video is {orig_size / 1048576:.1f}MB. Compressing (CPU)... (Video: {video_kbps}kbps, Audio: {audio_kbps}kbps)")
            
            final_path = os.path.join(YT_VIDEO_DIR, f"compressed_{uuid.uuid4().hex}.mp4")
            
            # Reverted to libx264 with strict maxrate buffers to prevent payload overshoots
            cmd = [
                "ffmpeg", "-y", "-nostdin",
                "-i", orig_path,
                "-c:v", "libx264",                  
                "-preset", "veryfast",
                "-b:v", f"{video_kbps}k",
                "-maxrate", f"{video_kbps}k",
                "-bufsize", f"{video_kbps * 2}k",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", 
                "-b:a", f"{audio_kbps}k", 
                final_path
            ]
            
            proc = await asyncio.create_subprocess_exec(
                *cmd, 
                stdout=asyncio.subprocess.DEVNULL, 
                stderr=asyncio.subprocess.PIPE
            )
            
            time_regex = re.compile(r"time=(\d{2}):(\d{2}):(\d{2}\.\d+)")
            speed_regex = re.compile(r"speed=\s*([\d\.]+)x")
            size_regex = re.compile(r"size=\s*(\d+)kB")
            
            last_ff_update = 0
            
            while True:
                line = await proc.stderr.readline()
                if not line:
                    break
                line_str = line.decode('utf-8', errors='ignore')
                
                t_match = time_regex.search(line_str)
                s_match = speed_regex.search(line_str)
                sz_match = size_regex.search(line_str)
                
                if t_match:
                    now = time.monotonic()
                    if now - last_ff_update > 1.2:
                        last_ff_update = now
                        h, m, s = t_match.groups()
                        curr_time = int(h) * 3600 + int(m) * 60 + float(s)
                        pct = min(100.0, (curr_time / max(duration, 1)) * 100)
                        
                        speed_str = f"{s_match.group(1)}x" if s_match else "1.0x"
                        size_mb = int(sz_match.group(1)) / 1024 if sz_match else 0
                        
                        eta_sec = (duration - curr_time) / float(s_match.group(1)) if (s_match and float(s_match.group(1)) > 0 and duration) else 0
                        eta_str = _fmt_time(eta_sec)

                        stats = get_sys_stats_string()
                        await status.update(f"🔄 Transcoding (CPU)... {_bar(pct)} {pct:.0f}% | ⚡ {speed_str} | ⏳ ETA: {eta_str} | 📦 {size_mb:.1f}MB\n{stats}")
            
            await proc.wait()
            if proc.returncode != 0:
                raise MusicError("FFmpeg failed to compress the video.")
                
            await status.update(f"✅ Compression Finished {_bar(100)} 100%", force=True)

            try:
                os.remove(orig_path)
                compressed = True
            except OSError:
                pass
                
        # --- 5. Upload to Discord & Register Cache ---
        await status.step("Uploading to Discord...")
        safe_name = _safe_filename(c['filename']) + ".mp4"
        file = discord.File(final_path, filename=safe_name)
        
        await ctx.send(content=f"🎥 **{c['filename']}**", file=file)
        _cache_put(cache_key, final_path)
        await status.finish("Video downloaded and sent successfully!")

    finally:
        if orig_path and final_path and orig_path != final_path and os.path.exists(orig_path):
            try:
                os.remove(orig_path)
            except OSError:
                pass

# ----------------- Command Groups -----------------

@bot.command()
async def help(ctx):
    """Displays the custom help menu."""
    embed = discord.Embed(
        title="🔍 Bot Commands",
        description="Here is everything I can do. For media types, use `image`, `gif`, or `video`.",
        color=discord.Color.blurple()
    )
    
    # --- Immich Commands ---
    immich_help = (
        "`!search <type> <query>` - ML Smart Search\n"
        "`!ocr <type> <query>` - Search by text inside media\n"
        "`!filename <type> <query>` - Search by file name\n"
        "`!desc <type> <query>` - Search by description"
    )
    embed.add_field(name="📸 Immich Search", value=immich_help, inline=False)

    # --- LLM Commands ---
    llm_help = (
        "`!llm start` / `!llm stop` - Toggles AI listening in the channel\n"
        "`!llm purge` - Clears memory context\n"
        "`!llm models` - Lists all available models\n"
        "`!llm switch <name>` - Switches models & unloads old one from RAM"
    )
    embed.add_field(name="🧠 AI Chat Mode", value=llm_help, inline=False)
    
    # --- Music & Media Commands ---
    music_help = (
        "`!music <track>` - Sends track as voice msg (Soulseek)\n"
        "`!music play <track>` - Plays in Voice Channel (Soulseek)\n"
        "`!ytmusic <track>` - Sends track as voice msg (YouTube)\n"
        "`!ytmusic play <track>` - Plays in Voice Channel (YouTube)\n"
        "`!yt <video>` - Downloads, compresses & uploads a YouTube video\n"
        "`!music np` / `!music q` - Shows current track / queue\n"
        "`!music skip` / `!music stop` - Skip track / stop completely\n"
        "`!music loop <n>` - Loops track\n"
        "`!playback speed <0.1-10>` - Live playback speed change"
    )
    embed.add_field(name="🎵 Music & Video", value=music_help, inline=False)
    
    # --- Utils ---
    utils_help = (
        "`!ping` - Check bot latency\n"
        "`!taskmanager` - Live 10s Server Hardware Monitor\n"
        "`!cache purge` - Deletes all downloaded media and frees space"
    )
    embed.add_field(name="🏓 Utility", value=utils_help, inline=False)
    
    await ctx.send(embed=embed)


@bot.command()
async def ping(ctx):
    """Ping command to check Discord API latency."""
    latency = round(bot.latency * 1000)
    await ctx.send(f"🏓 Pong! Latency is {latency}ms.")

@bot.command()
async def taskmanager(ctx):
    """Displays a live-updating system resource monitor for 10 seconds."""
    msg = None
    
    # Prime the CPU/Power metrics so the first read isn't 0
    psutil.cpu_percent(interval=None)
    get_sys_telemetry()
    await asyncio.sleep(0.5)

    for _ in range(20):
        stats = get_sys_telemetry()
        
        embed = discord.Embed(title="📊 Live System Task Manager", color=discord.Color.brand_green())
        
        # CPU Field
        cpu = stats["cpu"]
        embed.add_field(name="🖥️ Core i3-10100", 
                        value=f"**Usage:** {cpu['usage']}\n**Freq:** {cpu['freq']}\n**Temp:** {cpu['temp']}\n**Power:** {cpu['watts']}", 
                        inline=True)
        
        # GPU Field
        igpu = stats["igpu"]
        embed.add_field(name="🎮 UHD Graphics 630", 
                        value=f"**Usage:** {igpu['usage']}\n**Freq:** {igpu['freq']}\n**Temp:** {igpu['temp']}\n**Power:** {igpu['watts']}", 
                        inline=True)
        
        # Empty inline field for spacing
        embed.add_field(name="\u200b", value="\u200b", inline=True)
        
        # Memory Field
        ram = stats["ram"]
        embed.add_field(name="🧠 System RAM", 
                        value=f"**Usage:** {ram['usage']}\n**Amount:** {ram['text']}", 
                        inline=True)
        
        # VRAM Field
        embed.add_field(name="📼 Video RAM", 
                        value=f"**Usage:** Dynamic Allocation\n**Type:** {igpu['vram']}", 
                        inline=True)

        # Empty inline field for spacing
        embed.add_field(name="\u200b", value="\u200b", inline=True)

        embed.set_footer(text="Live updating... (Stops after 20s)")

        if msg is None:
            msg = await ctx.send(embed=embed)
        else:
            try:
                await msg.edit(embed=embed)
            except discord.HTTPException:
                pass
            
        await asyncio.sleep(1.2)

    # Final edit to indicate it's stopped
    embed.set_footer(text="🛑 Live updates stopped.")
    embed.color = discord.Color.dark_gray()
    await msg.edit(embed=embed)

# --- !search (Context/ML) ---
@bot.group(invoke_without_command=True)
async def search(ctx, media_type: str, *, query: str):
    await execute_search(ctx, "/api/search/smart", {"query": query}, "Context", media_type, query)

# --- !ocr ---
@bot.group(invoke_without_command=True)
async def ocr(ctx, media_type: str, *, query: str):
    payload = {"ocr": query}
    await execute_search(ctx, "/api/search/metadata", payload, "OCR", media_type, query)

# --- !filename ---
@bot.group(invoke_without_command=True)
async def filename(ctx, media_type: str, *, query: str):
    payload = {"originalFileName": query}
    await execute_search(ctx, "/api/search/metadata", payload, "File Name", media_type, query)

# --- !desc ---
@bot.group(invoke_without_command=True)
async def desc(ctx, media_type: str, *, query: str):
    payload = {"description": query}
    await execute_search(ctx, "/api/search/metadata", payload, "Description", media_type, query)

# ----------------- LLM Commands -----------------
@bot.group(invoke_without_command=True)
async def llm(ctx):
    """Base command for LLM."""
    await ctx.send("Use `!llm start` to enable chat mode, or `!llm help` for more info.")

@llm.command()
async def start(ctx):
    llm_active_channels.add(ctx.channel.id)
    current_prompt = AVAILABLE_MODELS[ACTIVE_MODEL_KEY]["prompt"]
    channel_histories[ctx.channel.id] = [{"role": "system", "content": current_prompt}]
    await ctx.send(f"🟢 LLM Active. Currently running: **{ACTIVE_MODEL_KEY}**")

@llm.command()
async def stop(ctx):
    if ctx.channel.id in llm_active_channels:
        llm_active_channels.discard(ctx.channel.id)
        if ctx.channel.id in channel_histories:
            del channel_histories[ctx.channel.id]
        await ctx.send("🛑 LLM Stopped.")

@llm.command(aliases=['reset', 'contextpurge'])
async def purge(ctx):
    if ctx.channel.id in llm_active_channels:
        current_prompt = AVAILABLE_MODELS[ACTIVE_MODEL_KEY]["prompt"]
        channel_histories[ctx.channel.id] = [{"role": "system", "content": current_prompt}]
        await ctx.send("🧹 Memory wiped. Context is fresh.")

@llm.command()
async def models(ctx):
    """Lists all available models."""
    embed = discord.Embed(title="🧠 Available LLM Models", color=discord.Color.green())
    for key, data in AVAILABLE_MODELS.items():
        status = "🟢 (Active)" if key == ACTIVE_MODEL_KEY else "⚪"
        embed.add_field(name=f"{status} {key}", 
                        value=f"**Ollama Model:** `{data['model_name']}`\n**Prompt:** *{data['prompt'][:60]}...*", 
                        inline=False)
    await ctx.send(embed=embed)

@llm.command()
async def switch(ctx, model_key: str):
    """Switches active model and UNLOADS the previous one from RAM."""
    global ACTIVE_MODEL_KEY, LLM_MODEL
    model_key = model_key.lower()
    
    if model_key not in AVAILABLE_MODELS:
        await ctx.send(f"❌ Model `{model_key}` not found. Use `!llm models` to see available options.")
        return
        
    old_model = LLM_MODEL
    ACTIVE_MODEL_KEY = model_key
    LLM_MODEL = AVAILABLE_MODELS[model_key]["model_name"]
    
    # UNLOAD the previous model from RAM to save server memory
    if old_model != LLM_MODEL:
        await ctx.send(f"🧹 Unloading `{old_model}` from server RAM...")
        await unload_ollama_model(old_model)
    
    # Update active channels with the new prompt and wipe memory
    new_prompt = AVAILABLE_MODELS[model_key]["prompt"]
    for channel_id in llm_active_channels:
        if channel_id in channel_histories:
            channel_histories[channel_id] = [{"role": "system", "content": new_prompt}]
            
    await ctx.send(f"🔄 **Switched!** Now using `{model_key}` (`{LLM_MODEL}`). Memory has been wiped for safety.")

@llm.command()
async def addmodel(ctx, name: str, model_path: str, *, prompt: str):
    """Adds a new model on the fly."""
    AVAILABLE_MODELS[name.lower()] = {"model_name": model_path, "prompt": prompt}
    await ctx.send(f"✅ Successfully added `{name.lower()}` pointing to `{model_path}`.")
    await ctx.send("⚠️ *Note: Because models are kept strictly self-contained in memory, this model will disappear if the bot is restarted.*")
    
# ----------------- Music Commands -----------------
@bot.group(invoke_without_command=True)
async def music(ctx, *, query: str = None):
    """!music <track> → sends it as a voice message."""
    if not query:
        await ctx.send("Usage: `!music Artist - Title` or `!music play Artist - Title`")
        return
    await run_music(ctx, query, mode="file")

@music.command(name="play")
async def music_play(ctx, *, query: str = None):
    """!music play <track> → plays it in your voice channel (or queues it)."""
    if not query:
        await ctx.send("Usage: `!music play Artist - Title`")
        return
    await run_music(ctx, query, mode="play")


@music.command(name="stop")
async def music_stop(ctx):
    """Stops playback, clears the queue and leaves the voice channel."""
    player = music_players.get(ctx.guild.id) if ctx.guild else None
    if player:
        cur, dropped = player.current, list(player.queue)
        player.current = None                      # so the end-of-track callback does nothing
        player.queue.clear()
        if cur:
            await cur["status"].finish(f"Stopped **{cur['label']}**")
        for t in dropped:
            await t["status"].finish("Removed from the queue — playback stopped", ok=False)
    if ctx.voice_client:
        await ctx.voice_client.disconnect()
        await ctx.send("⏹️ Stopped.")
    else:
        await ctx.send("I'm not in a voice channel.")

@music.command(name="skip")
async def music_skip(ctx):
    """!music skip → skips the current track (any remaining loops are dropped)."""
    player = music_players.get(ctx.guild.id) if ctx.guild else None
    vc = ctx.voice_client
    if not (player and player.current and vc and (vc.is_playing() or vc.is_paused())):
        await ctx.send("Nothing is playing right now.")
        return
    track = player.current
    bypassed = track["loops_left"]
    track["loops_left"] = 0                        # bypass loops entirely
    track["skipped"] = True                        # end-of-track callback reports "Skipped" and moves on
    vc.stop()

    msg = f"⏭️ Skipped **{discord.utils.escape_markdown(track['label'])}**"
    if bypassed:
        msg += f" (dropped {bypassed} remaining loop{'s' if bypassed != 1 else ''})"
    if player.queue:
        msg += f"\nUp next: **{discord.utils.escape_markdown(player.queue[0]['label'])}**"
    else:
        msg += "\nThe queue is empty — leaving the voice channel."
    await ctx.send(msg)

@music.command(name="loop")
async def music_loop(ctx, count: str = None):
    """!music loop <n> → repeats the current track n more times (0 clears the loop)."""
    player = music_players.get(ctx.guild.id) if ctx.guild else None
    vc = ctx.voice_client
    if not (player and player.current and vc and (vc.is_playing() or vc.is_paused())):
        await ctx.send("Nothing is playing right now.")
        return
    try:
        n = int(count)
    except (TypeError, ValueError):
        await ctx.send("Usage: `!music loop <number>` (e.g. `!music loop 3`, `!music loop 0` to clear)")
        return
    if n < 0:
        await ctx.send("Loop count must be `0` or higher.")
        return

    track = player.current
    track["loops_left"] = n
    await track["status"].update(_playing_text(track, vc), force=True)
    label = discord.utils.escape_markdown(track["label"])
    if n == 0:
        await ctx.send(f"🔁 Loop cleared for **{label}**.")
    else:
        await ctx.send(f"🔁 **{label}** will repeat **{n}** more time{'s' if n != 1 else ''} "
                       f"({n + 1} plays total). `!music skip` ends it early.")

@music.command(name="np", aliases=["nowplaying"])
async def music_np(ctx):
    """Shows the currently playing track."""
    player = music_players.get(ctx.guild.id) if ctx.guild else None
    vc = ctx.voice_client
    if not (player and player.current and vc and (vc.is_playing() or vc.is_paused())):
        await ctx.send("Nothing is playing right now.")
        return
    await ctx.send(embed=build_np_embed(player))

@music.command(name="q", aliases=["queue"])
async def music_queue(ctx):
    """Shows the current queue."""
    player = music_players.get(ctx.guild.id) if ctx.guild else None
    if not player or not (player.current or player.queue):
        await ctx.send("The queue is empty.")
        return
    await ctx.send(embed=build_queue_embed(player))

@bot.group(invoke_without_command=True)
async def cache(ctx):
    """Base command for cache management."""
    await ctx.send("Usage: `!cache purge` to clear all downloaded media.")

@cache.command(name="purge")
async def cache_purge(ctx):
    """!cache purge → deletes everything in the downloads/cache folder."""
    loop = asyncio.get_running_loop()
    files_deleted, size, errors = await loop.run_in_executor(None, _purge_cache_dir)
    _cache_clear()

    lost = 0                                               
    for p in music_players.values():
        keep = []
        for t in p.queue:
            if os.path.isfile(t["path"]):
                keep.append(t)
            else:
                lost += 1
                await t["status"].finish("Removed from the queue — cache was purged", ok=False)
        p.queue[:] = keep

    size_gb = size / (1024 * 1024 * 1024)
    msg = f"🗑️ Cache purged — **{files_deleted} files** and **{size_gb:.2f} GB** freed up."
    
    if lost:
        msg += f"\n⚠️ {lost} queued track{'s' if lost != 1 else ''} removed (file deleted)."
    if errors:
        msg += f"\n⚠️ {errors} item{'s' if errors != 1 else ''} couldn't be deleted (check folder permissions)."
        
    await ctx.send(msg)


@bot.group(invoke_without_command=True)
async def ytmusic(ctx, *, query: str = None):
    """!ytmusic <track> → downloads from YouTube and sends it as a voice message."""
    if not query:
        await ctx.send("Usage: `!ytmusic <search query or URL>` or `!ytmusic play <query>`")
        return
    await run_yt_music(ctx, query, mode="file")

@ytmusic.command(name="play")
async def ytmusic_play(ctx, *, query: str = None):
    """!ytmusic play <track> → plays it in your voice channel (or queues it)."""
    if not query:
        await ctx.send("Usage: `!ytmusic play <search query or URL>`")
        return
    await run_yt_music(ctx, query, mode="play")

@bot.command()
async def yt(ctx, *, query: str = None):
    """!yt <query> → Searches, downloads, compresses, and uploads a YouTube video."""
    if not query:
        await ctx.send("Usage: `!yt <search query or URL>`")
        return
        
    status = MusicStatus(ctx, query, "video")
    if _music_sem.locked():
        await status.step("Waiting for another media request to finish…")
        
    async with _music_sem:
        try:
            await run_yt_video_download(ctx, query, status)
        except MusicError as e:
            await status.finish(str(e), ok=False)
        except Exception as e:
            await status.finish(f"Something broke: `{e}`", ok=False)

# ----------------- Playback Commands -----------------
@bot.group(invoke_without_command=True)
async def playback(ctx):
    """Base command for live playback controls."""
    await ctx.send(f"Usage: `!playback speed <{MUSIC_SPEED_MIN:g}-{MUSIC_SPEED_MAX:g}>` "
                   "(e.g. `!playback speed 1.5`, `!playback speed 1` for normal)")

@playback.command(name="speed")
async def playback_speed(ctx, value: str = None):
    """!playback speed <0.1-10> → changes the speed of the current (and later) tracks live."""
    player = music_players.get(ctx.guild.id) if ctx.guild else None
    vc = ctx.voice_client
    if not (player and player.current and vc and (vc.is_playing() or vc.is_paused())):
        await ctx.send("Nothing is playing right now.")
        return
    if value is None:
        await ctx.send(f"⏩ Current speed is **{_fmt_speed(player.speed)}**. "
                       f"Use `!playback speed <{MUSIC_SPEED_MIN:g}-{MUSIC_SPEED_MAX:g}>` to change it.")
        return
    try:
        speed = float(value.lower().rstrip("x"))
    except ValueError:
        await ctx.send(f"Usage: `!playback speed <{MUSIC_SPEED_MIN:g}-{MUSIC_SPEED_MAX:g}>`")
        return
    if not (MUSIC_SPEED_MIN <= speed <= MUSIC_SPEED_MAX):          # also rejects nan
        await ctx.send(f"Speed must be between **{MUSIC_SPEED_MIN:g}x** and **{MUSIC_SPEED_MAX:g}x**.")
        return
    if speed == player.speed:
        await ctx.send(f"Already playing at **{_fmt_speed(speed)}**.")
        return

    track = player.current
    pos = _track_position(track)
    player.speed = speed
    if track["duration"] and pos >= track["duration"] - 1:         # basically finished → next play uses it
        await ctx.send(f"⏩ Speed set to **{_fmt_speed(speed)}** (applies from the next play).")
        return
    try:
        await _restart_current(vc, player, pos)                    # resume from the same spot
    except Exception as e:
        await ctx.send(f"❌ Couldn't change speed: `{e}`")
        return
    await ctx.send(f"⏩ Playback speed set to **{_fmt_speed(speed)}** (resumed at `{_fmt_time(pos)}`).")

# ----------------- Message Listener -----------------
@bot.event
async def on_message(message):
    # Ignore the bot's own messages
    if message.author == bot.user:
        return

    # Process normal commands (!search, !llm start, etc) FIRST
    await bot.process_commands(message)

    # If it's a command, do not pass it to the LLM
    if message.content.startswith(bot.command_prefix):
        return

    # If the channel is in LLM mode, respond to standard text
    if message.channel.id in llm_active_channels:

        # Ignore empty messages (like if someone uploads an image with no text)
        if not message.content.strip():
            return

        async with message.channel.typing():
            try:
                # 1. Format using brackets <Name> instead of Name: to stop script bleed
                clean_user_msg = f"<{message.author.display_name}> {message.content.strip()}"
                channel_histories[message.channel.id].append(
                    {"role": "user", "content": clean_user_msg}
                )

                if len(channel_histories[message.channel.id]) > 5:
                    channel_histories[message.channel.id] = [channel_histories[message.channel.id][0]] + channel_histories[message.channel.id][-4:]

                messages_for_api = list(channel_histories[message.channel.id])

                if CHAOS_PROMPTS and random.random() < CHAOS_CHANCE:
                    random_instruction = random.choice(CHAOS_PROMPTS)
                    injection_note = f"[System note: {random_instruction}]"
                else:
                    current_persona = channel_histories[message.channel.id][0]["content"]
                    injection_note = f"[System note: Stay deeply in character. {current_persona}]"

                messages_for_api[-1] = {
                    "role": "user",
                    "content": f"{clean_user_msg}\n\n{injection_note}"
                }

                # 2. PASS STOP TOKENS to physically prevent formatting hallucinations
                response = await llm_client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=messages_for_api,
                    max_tokens=250,
                    stop=["\n<"]
                )

                # 3. Strip whitespace to fix the random indentation issue
                reply = response.choices[0].message.content.strip()

                # 4. Fallback failsafe: sever the message if it types a name
                for prefix in STOP_PREFIXES:
                    idx = reply.lower().find(prefix)
                    if idx != -1:
                        reply = reply[:idx].strip()

                if not reply:
                    reply = "..."

                channel_histories[message.channel.id].append(
                    {"role": "assistant", "content": reply}
                )

                if len(reply) > 2000:
                    for i in range(0, len(reply), 2000):
                        await message.channel.send(reply[i:i+2000])
                else:
                    await message.reply(reply)

            except Exception as e:
                await message.channel.send(f"❌ Error communicating with local AI: {e}")


@bot.event
async def on_ready():
    print(f'Bot is online! Logged in as {bot.user}')

if __name__ == "__main__":
    if not DISCORD_BOT_TOKEN:
        raise SystemExit("DISCORD_BOT_TOKEN isn't set - copy .env.example to .env and fill it in.")
    bot.run(DISCORD_BOT_TOKEN)
