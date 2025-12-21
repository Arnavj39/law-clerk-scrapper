#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Law Clerk headshot harvester (Bing only; no Selenium/WebDriver).

Key fixes:
• Hyphen/diacritic-safe name gate.
• High-trust host checked via (purl or murl); smarter title budget usage.
• Person folder created only after _1.jpg is secured.
• Staged fallbacks if strict filters find too few images (relaxed name gate,
  drop law/clerk for low-trust only as needed, and soften face/quality gates).
• Better pHash de-dupe seeding even when _1 face crop fails.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Silence noisy pkg_resources warning from face_recognition_models only
# ──────────────────────────────────────────────────────────────────────────────

import warnings
warnings.filterwarnings(
    "ignore",
    message=r".*pkg_resources is deprecated.*",
    category=UserWarning,
    module=r"face_recognition_models(\.|$)",
)

# ──────────────────────────────────────────────────────────────────────────────
# Imports & globals
# ──────────────────────────────────────────────────────────────────────────────

import os, sys, csv, time, json, math, random, shutil, re
from io import BytesIO
from typing import List, Dict, Tuple, Optional, Set
from urllib.parse import quote, urlparse, urlunparse

import requests
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from bs4 import BeautifulSoup
import face_recognition  # HOG fallback

# Optional libs
import cv2
_HAS_CV2 = hasattr(cv2, "cvtColor") and hasattr(cv2, "dct") and hasattr(cv2, "inRange")
try:
    from retinaface import RetinaFace
    _HAS_RETINAFACE = True
except Exception:
    _HAS_RETINAFACE = False
if not _HAS_CV2:
    _HAS_RETINAFACE = False

_HAS_DEEPFACE = False
try:
    from deepface import DeepFace
    _HAS_DEEPFACE = True
except Exception:
    _HAS_DEEPFACE = False

try:
    from gender_guesser import detector as gender
    _GENDER = gender.Detector(case_sensitive=False)
except Exception:
    _GENDER = None

# ──────────────────────────────────────────────────────────────────────────────
# Tunables
# ──────────────────────────────────────────────────────────────────────────────

REQUEST_TIMEOUT = 18
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5_0) AppleWebKit/605.1.15 "
                   "(KHTML, like Gecko) Version/18.0 Safari/605.1.15"),
    "Accept-Language": "en-US,en;q=0.9",
}
SLEEP_MIN, SLEEP_MAX = 0.25, 0.55
KEEP_IMAGES_TARGET = 7
IMAGES_PER_PAGE = 35
MAX_PAGES = 4
CONTEXT_PAD_FRAC = 0.70

SOURCE_WEIGHTS = {
    "Gov": 7, "Edu": 6, "Martindale-Hubbell": 6, "LinkedIn": 5, "Org": 5,
    "Firm": 4, "Professional": 3, "News": 3, "Other": 1
}

# Column header candidates
POSSIBLE_FIRST = ["first name", "first"]
POSSIBLE_MIDDLE = ["middle name", "middle initial", "middle initial (if present)", "middle"]
POSSIBLE_LAST = ["last name", "last"]
POSSIBLE_SUFFIX = ["suffix"]
POSSIBLE_LINKEDIN = ["linkedin", "linkedin url", "linkdeln", "linkden", "linkdin"]
POSSIBLE_EDU = ["education", "school", "law school"]
POSSIBLE_JUDGE = [
    "judge name (unique for each person)", "judge name", "judge", "chambers", "chambers / judge",
    "organization name (child)","organization name (intermediate)","organization name (parent)"
]

# Real-face filtering thresholds (strict)
MIN_RETINAFACE_SCORE = 0.90
MIN_FACE_AREA_FRAC = 0.03
MAX_FACE_AREA_FRAC = 0.70
MIN_ENTROPY_BITS = 4.0
MIN_SKIN_FRAC = 0.02
PHASH_HAMMING_NEAR = 8

# URL tokens indicating artwork / non-photo
ART_URL_TOKENS = [
    "icon","saint","byzantine","mosaic","painting","illustration","clipart","vector","avatar",
    "emoji","cartoon","drawing","orthodox","fresco","statue","sculpture"
]

# Image safety limits
MAX_IMAGE_BYTES = 3_500_000
MAX_IMAGE_PIXELS = 12_000_000
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

# Title fetch budget (per person)
TITLE_FETCH_BUDGET = 4
_title_budget_remain = 0

LAW_KEYWORDS = ["law","clerk","court","judge","attorney","firm","chambers"]

# ──────────────────────────────────────────────────────────────────────────────
# Small utils
# ──────────────────────────────────────────────────────────────────────────────

def _safe(x) -> str:
    if x is None: return ""
    if isinstance(x, float) and math.isnan(x): return ""
    return str(x).strip()

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _norm_host(u: str) -> str:
    try: return urlparse(u).netloc.lower()
    except Exception: return ""

def _canon_url(u: str) -> str:
    u = (u or "").strip()
    if not u: return ""
    try:
        p = urlparse(u)
        path = p.path
        path = re.sub(r"(?i)(?:\b\d{2,4}x\d{2,4}\b|[_-]\d{2,4}(?=\.))", "", path)
        return (p.netloc.lower() + path).lower()
    except Exception:
        return u.split("?", 1)[0].lower()

def _domain_category(u: str) -> str:
    h = _norm_host(u)
    if not h: return "Other"
    if "martindale" in h: return "Martindale-Hubbell"
    if "linkedin.com" in h or "licdn.com" in h: return "LinkedIn"
    if h.endswith(".gov"): return "Gov"
    if h.endswith(".edu"): return "Edu"
    if h.endswith(".org"): return "Org"
    if any(tok in h for tok in ("law","llp","attorney","firm","barrister","chambers")): return "Firm"
    if any(tok in h for tok in ("news","times","wsj","reuters","bloomberg","guardian")): return "News"
    return "Other"

def _is_high_trust_host(u: str) -> bool:
    return _domain_category(u) in {
        "LinkedIn","Gov","Edu","Firm","Martindale-Hubbell","Org","Professional","News"
    }

def _middle_initial(middle: str) -> str:
    m = (middle or "").strip()
    if not m: return ""
    ch = next((c for c in m if c.isalpha()), "")
    return f"{ch.upper()}." if ch else ""

def _clean_token(s: str) -> str:
    s = "".join(c for c in s if c.isalnum() or c in (" ", "-", "_", "'"))
    s = s.replace("'", "")
    return "_".join(s.split())

def _expand_box(box: Tuple[int,int,int,int], W: int, H: int, pad_frac: float) -> Tuple[int,int,int,int]:
    """
    Expand a (t, r, b, l) box by pad_frac in both width and height, clamped to image bounds.
    pad_frac=0.30 means new width = 1.30 * old width, same for height.
    """
    t, r, b, l = box
    w = max(1, r - l)
    h = max(1, b - t)
    cx = l + w / 2.0
    cy = t + h / 2.0
    w2 = w * (1.0 + pad_frac)
    h2 = h * (1.0 + pad_frac)
    l2 = int(round(cx - w2 / 2.0))
    r2 = int(round(cx + w2 / 2.0))
    t2 = int(round(cy - h2 / 2.0))
    b2 = int(round(cy + h2 / 2.0))
    l2 = max(0, min(l2, W - 1))
    r2 = max(1, min(r2, W))
    t2 = max(0, min(t2, H - 1))
    b2 = max(1, min(b2, H))
    if r2 <= l2 or b2 <= t2:
        return (t, r, b, l)
    return (t2, r2, b2, l2)

def _name_blocks(first: str, middle: str, last: str, suffix: str) -> Tuple[str,str,str,str]:
    return _clean_token(first), _clean_token(_middle_initial(middle)), _clean_token(last), _clean_token(suffix)

def file_prefix(first: str, middle: str, last: str, suffix: str) -> str:
    f,m,l,s = _name_blocks(first, middle, last, suffix)
    return "_".join([f,m,l,s]) + "_"

def folder_name(first: str, middle: str, last: str, suffix: str) -> str:
    return file_prefix(first, middle, last, suffix)

def sleep_jitter():
    time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))

def reset_title_budget(n: int = TITLE_FETCH_BUDGET):
    global _title_budget_remain
    _title_budget_remain = n

# ──────────────────────────────────────────────────────────────────────────────
# HTTP
# ──────────────────────────────────────────────────────────────────────────────

SESSION = requests.Session()
SESSION.headers.update(HEADERS)

def http_get(url: str, allow_404: bool = False) -> Optional[requests.Response]:
    tries = 3
    for attempt in range(tries):
        try:
            r = SESSION.get(url, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200: return r
            if allow_404 and r.status_code == 404: return r
            if r.status_code in (429,500,502,503,504):
                time.sleep(0.8*(attempt+1))
            else:
                return None
        except Exception:
            time.sleep(0.5*(attempt+1))
    return None

def download_bytes_limited(url: str, max_bytes: int = MAX_IMAGE_BYTES) -> Optional[bytes]:
    try:
        try:
            hr = SESSION.head(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
            if hr is not None:
                clen = hr.headers.get("Content-Length") or hr.headers.get("content-length")
                if clen and clen.isdigit() and int(clen) > max_bytes:
                    return None
        except Exception:
            pass
        r = SESSION.get(url, timeout=REQUEST_TIMEOUT, stream=True)
        if r.status_code != 200: return None
        buf = BytesIO(); read = 0
        for chunk in r.iter_content(chunk_size=65536):
            if not chunk: break
            buf.write(chunk); read += len(chunk)
            if read > max_bytes: return None
        return buf.getvalue()
    except Exception:
        return None

def download_tile_image(t: Dict, max_bytes: int = MAX_IMAGE_BYTES) -> Optional[bytes]:
    """
    Prefer Bing thumbnail (turl) to keep bytes small; fall back to original (murl).
    max_bytes lets us relax the size cap in last-resort salvage.
    """
    for key in ("turl", "murl"):
        url = t.get(key)
        if not url:
            continue
        b = download_bytes_limited(url, max_bytes=max_bytes)
        if b:
            return b
    return None

# ──────────────────────────────────────────────────────────────────────────────
# Bing Images helpers (with paging)
# ──────────────────────────────────────────────────────────────────────────────

def _parse_images_tiles(html: str) -> List[Dict]:
    out = []
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a", class_="iusc"):
        try: meta = json.loads(a.get("m", "{}"))
        except Exception: meta = {}
        murl = (meta or {}).get("murl") or ""
        turl = (meta or {}).get("turl") or ""
        purl = (meta or {}).get("purl") or ""
        if murl: out.append({"murl": murl, "turl": turl, "purl": purl})
    return out

def bing_images_tiles_across_pages(query: str, want: int, max_pages: int = MAX_PAGES) -> List[Dict]:
    tiles: List[Dict] = []
    q = quote(query)
    url0 = f"https://www.bing.com/images/search?q={q}&form=HDRSC2&count={IMAGES_PER_PAGE}"
    r = http_get(url0)
    if r: tiles.extend(_parse_images_tiles(r.text))
    offset = IMAGES_PER_PAGE + 1; page = 2
    while len(tiles) < want and page <= max_pages:
        url_async = (f"https://www.bing.com/images/async?q={q}&first={offset}"
                     f"&count={IMAGES_PER_PAGE}&relp={IMAGES_PER_PAGE}&scenario=ImageBasicHover")
        r = http_get(url_async)
        if not r: break
        got = _parse_images_tiles(r.text)
        if not got: break
        tiles.extend(got); offset += IMAGES_PER_PAGE; page += 1
        sleep_jitter()
    return tiles

# ──────────────────────────────────────────────────────────────────────────────
# LinkedIn URL normalization + fuzzy match
# ──────────────────────────────────────────────────────────────────────────────

def normalize_linkedin_url(u: str) -> str:
    if not u: return ""
    try:
        p = urlparse(u.strip())
        scheme = "https"; netloc = p.netloc.lower().replace("www.", "")
        path = p.path.rstrip("/")
        return urlunparse((scheme, netloc, path, "", "", ""))
    except Exception:
        return u.strip().lower().rstrip("/")

def linkedin_similarity(a: str, b: str) -> float:
    na, nb = normalize_linkedin_url(a), normalize_linkedin_url(b)
    if not na or not nb: return 0.0
    if na == nb: return 1.0
    if na.startswith(nb) or nb.startswith(na): return 0.8
    try:
        last_a = [seg for seg in urlparse(na).path.split("/") if seg][-1]
        last_b = [seg for seg in urlparse(nb).path.split("/") if seg][-1]
        if last_a and last_b and last_a.lower() == last_b.lower(): return 0.6
    except Exception:
        pass
    return 0.0

# ──────────────────────────────────────────────────────────────────────────────
# Name matching helpers (hyphen/diacritic safe) + title gating
# ──────────────────────────────────────────────────────────────────────────────

TITLE_CACHE: Dict[str, str] = {}

def _fetch_title(url: str) -> str:
    if not url: return ""
    if url in TITLE_CACHE: return TITLE_CACHE[url]
    title = ""
    r = http_get(url)
    if r and ("html" in (r.headers.get("Content-Type", "") or "").lower()):
        try:
            soup = BeautifulSoup(r.text, "html.parser")
            title = (soup.title.string or "").strip()
        except Exception:
            title = ""
    TITLE_CACHE[url] = title
    return title

def _normalize_for_match(s: str) -> str:
    # Lowercase, strip diacritics, replace non-alnum with single spaces
    s = s.lower()
    try:
        import unicodedata
        s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    except Exception:
        pass
    return re.sub(r"[^a-z0-9]+", " ", (s or "")).strip()

def _tokens(s: str) -> List[str]:
    return [t for t in _normalize_for_match(s).split() if t]

def _name_match(text: str, first: str, last: str, mode: str = "strict") -> bool:
    """
    strict  : require first AND all last-name tokens (hyphens split) to appear
    relaxed : require first AND any last-name token
    loose   : require first OR last (used only for clearly high-trust fallbacks)
    """
    toks = set(_tokens(text))
    f = _tokens(first); l = _tokens(last)
    f_ok = any(t in toks for t in f) if f else False
    if mode == "strict":
        l_ok = all(t in toks for t in l) if l else False
        return f_ok and l_ok
    if mode == "relaxed":
        l_ok = any(t in toks for t in l) if l else False
        return f_ok and l_ok
    # loose
    l_ok_any = any(t in toks for t in l) if l else False
    return f_ok or l_ok_any

def _tile_about_person(
    t: Dict,
    first: str,
    last: str,
    required_terms: Optional[List[str]] = None,
    name_mode: str = "strict",
) -> bool:
    """
    Accept a tile if (URL OR limited-budget title on high-trust host) contains the name,
    with the chosen name_mode. For low-trust hosts ("Other"), also require any required_term.
    """
    global _title_budget_remain

    purl = (t.get("purl") or t.get("murl") or "")
    murl = t.get("murl") or ""
    cat = _domain_category(purl or murl)
    high_trust = cat != "Other"

    # quick URL check (works even when hyphens are normalized)
    url_name_ok = _name_match(purl, first, last, mode=name_mode)

    title = ""
    if not url_name_ok and high_trust and _title_budget_remain > 0 and t.get("purl"):
        title = _fetch_title(t["purl"])
        _title_budget_remain -= 1

    # name_ok via URL or (if fetched) title
    name_ok = url_name_ok or _name_match(title, first, last, mode=name_mode)

    # Extra allowance: licdn.com images with no purl are treated as high-trust;
    # for them, in the most-relaxed pass we allow 'loose' match implicitly.
    if not name_ok and not t.get("purl") and _domain_category(murl) == "LinkedIn":
        name_ok = (name_mode == "loose")

    if not name_ok:
        return False

    if required_terms:
        hay = (purl + " " + title).lower()
        if cat == "Other" and not any(term.lower() in hay for term in required_terms):
            return False

    return True

def _looks_like_art_url(u: str) -> bool:
    u = (u or "").lower()
    return any(tok in u for tok in ART_URL_TOKENS)

# ──────────────────────────────────────────────────────────────────────────────
# Vision & filtering helpers (+ tunable relax parameters)
# ──────────────────────────────────────────────────────────────────────────────

def _image_entropy_bits(img: Image.Image) -> float:
    arr = np.asarray(img.convert("RGB"))
    H = 0.0
    for c in range(3):
        hist, _ = np.histogram(arr[..., c], bins=256, range=(0,255), density=True)
        p = hist + 1e-12
        H += -np.sum(p * np.log2(p))
    return H / 3.0

def _skin_fraction(img: Image.Image) -> float:
    try:
        if _HAS_CV2:
            bgr = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)
            ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
            mask = cv2.inRange(ycrcb, (0,133,77), (255,173,127))
            return float(np.count_nonzero(mask)) / (mask.size + 1e-9)
        else:
            ycbcr = img.convert("YCbCr")
            arr = np.asarray(ycbcr)
            Cb = arr[...,1]; Cr = arr[...,2]
            mask = (Cr >= 135) & (Cr <= 180) & (Cb >= 85) & (Cb <= 135)
            return float(mask.sum()) / mask.size
    except Exception:
        return 0.0

def _phash64(img: Image.Image) -> int:
    g = img.convert("L").resize((32,32), Image.BICUBIC)
    A = np.array(g, dtype=np.float32)
    if _HAS_CV2:
        dmtx = cv2.dct(A)
    else:
        try:
            from scipy.fftpack import dct as sp_dct
            dmtx = sp_dct(sp_dct(A, axis=0, norm='ortho'), axis=1, norm='ortho')
        except Exception:
            dmtx = np.fft.fft2(A).real
    block = dmtx[:8,:8]
    med = float(np.median(block[1:]))
    bits = (block > med).astype(np.uint64).flatten()
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return int(val)
def _hamming64(a: int, b: int) -> int:
    """
    Hamming distance between two 64-bit pHashes.
    Uses int.bit_count() when available; falls back to bin().count('1').
    """
    x = (a ^ b) & ((1 << 64) - 1)
    try:
        return x.bit_count()          # Py 3.8+ on most builds
    except AttributeError:
        return bin(x).count("1")

def _phash_from_bytes(img_bytes: bytes) -> Optional[int]:
    try:
        im = Image.open(BytesIO(img_bytes)).convert("RGB")
        return _phash64(im)
    except Exception:
        return None

def extract_real_face(
    img_bytes: bytes,
    murl: str = "",
    purl: str = "",
    *,
    min_area: float = MIN_FACE_AREA_FRAC,
    max_area: float = MAX_FACE_AREA_FRAC,
    min_entropy_bits: float = MIN_ENTROPY_BITS,
    min_skin_frac: float = MIN_SKIN_FRAC,
    skip_quality_checks: bool = False,
) -> Optional[Image.Image]:
    """Return a cropped PIL face only if it looks photo-real (filters artwork/statues)."""
    if _looks_like_art_url(murl) or _looks_like_art_url(purl):
        return None
    try:
        base_img = Image.open(BytesIO(img_bytes)).convert("RGB")
    except Exception:
        return None
    H, W = base_img.height, base_img.width
    if H < 80 or W < 80:
        return None

    box = None
    if _HAS_RETINAFACE and _HAS_CV2:
        bgr = cv2.cvtColor(np.asarray(base_img), cv2.COLOR_RGB2BGR)
        try:
            faces = RetinaFace.detect_faces(bgr)
        except Exception:
            faces = None
        if isinstance(faces, dict) and len(faces) > 0:
            best = None; best_score = -1.0
            for _, v in faces.items():
                score = float(v.get("score", 0.0) or 0.0)
                x1, y1, x2, y2 = v.get("facial_area", [0,0,0,0])
                if score > best_score:
                    best_score = score; best = (x1, y1, x2, y2)
            if best is not None and best_score >= MIN_RETINAFACE_SCORE:
                x1, y1, x2, y2 = [int(round(v)) for v in best]
                x1 = max(0, min(x1, W-1)); x2 = max(0, min(x2, W))
                y1 = max(0, min(y1, H-1)); y2 = max(0, min(y2, H))
                if x2 > x1 and y2 > y1:
                    area_frac = (x2-x1)*(y2-y1)/float(W*H)
                    if min_area <= area_frac <= max_area:
                        box = (y1, x2, y2, x1)

    if box is None:
        try:
            arr = np.asarray(base_img)
            boxes = face_recognition.face_locations(arr, model="hog")
        except Exception:
            boxes = []
        if not boxes:
            return None
        def area(b): return max(0,b[2]-b[0]) * max(0,b[1]-b[3])
        t, r, b, l = max(boxes, key=area)
        area_frac = max(0,(b-t))*max(0,(r-l))/float(W*H)
        if not (min_area <= area_frac <= max_area):
            return None
        box = (t,r,b,l)

    t, r, b, l = _expand_box(box, W, H, CONTEXT_PAD_FRAC)
    face = base_img.crop((l, t, r, b)).resize((256,256), Image.LANCZOS)

    if face.getbands() == ("P",):
        return None
    if not skip_quality_checks:
        if _image_entropy_bits(face) < min_entropy_bits:
            return None
        if _skin_fraction(face) < min_skin_frac:
            return None
    return face

# ──────────────────────────────────────────────────────────────────────────────
# Search helpers (first tile, crops, staged fallbacks)
# ──────────────────────────────────────────────────────────────────────────────

def first_valid_linkedin_tile(query: str, first: str, last: str, tiles_cache: Optional[List[Dict]] = None) -> Optional[Dict]:
    tiles = tiles_cache if tiles_cache is not None else bing_images_tiles_across_pages(query, want=50, max_pages=3)

    def try_pass(name_mode: str) -> Optional[Dict]:
        for t in tiles:
            purl = (t.get("purl") or "")
            if not _tile_about_person(t, first, last, name_mode=name_mode):
                continue
            murl = t.get("murl") or ""
            if not murl: continue
            if _looks_like_art_url(murl) or _looks_like_art_url(purl): continue
            # Prefer LinkedIn hosts in first sweep
            b = download_tile_image(t)
            if not b: continue
            face = extract_real_face(b, murl=murl, purl=purl or murl)
            if not face: continue
            return {"img_bytes": b, "murl": murl, "purl": purl or murl, "search_query": query}
        return None

    hit = try_pass("strict")
    if hit: return hit
    return try_pass("relaxed")

def first_downloadable_tile_face_checked(query: str, first: str, last: str) -> Optional[Dict]:
    tiles = bing_images_tiles_across_pages(query, want=40, max_pages=2)
    for mode in ("strict","relaxed","loose"):
        for t in tiles:
            if not _tile_about_person(t, first, last, name_mode=mode):
                continue
            b = download_tile_image(t)
            if not b: continue
            face_ok = extract_real_face(b, murl=t.get("murl",""), purl=t.get("purl","") or t.get("murl",""))
            if face_ok is None: continue
            return {"img_bytes": b, "murl": t.get("murl",""), "purl": t.get("purl","") or t.get("murl",""), "search_query": query}
    return None

def _collect_with_step(tiles: List[Dict], need: int, *, first: str, last: str,
                       name_mode: str, required_terms: Optional[List[str]],
                       seen_urls: Set[str], seen_hashes: Set[int],
                       relax_level: int) -> Tuple[List[Image.Image], List[Tuple[str,str,int]]]:
    """
    relax_level: 0 strict, 1 softened (wider face area), 2 softest (skip entropy/skin on high-trust)
    """
    imgs: List[Image.Image] = []
    meta: List[Tuple[str,str,int]] = []

    for i, t in enumerate(tiles, start=1):
        purl = t.get("purl",""); murl = t.get("murl","")
        cat = _domain_category(purl or murl)
        if not _tile_about_person(t, first, last, required_terms=required_terms, name_mode=name_mode):
            continue

        # de-dupe by canonical URL
        canon = "|".join([_canon_url(t.get('murl','')), _canon_url(t.get('turl','')), _canon_url(t.get('purl',''))]).strip("|")
        if canon and canon in seen_urls: continue

        b = download_tile_image(t)
        if not b: continue

        # face params by relax level
        min_area = MIN_FACE_AREA_FRAC if relax_level == 0 else 0.02
        max_area = MAX_FACE_AREA_FRAC if relax_level == 0 else 0.85
        skip_quality = (relax_level >= 2) and (_domain_category(purl or murl) != "Other")  # only skip on high-trust

        face = extract_real_face(
            b, murl=murl, purl=purl or murl,
            min_area=min_area, max_area=max_area,
            min_entropy_bits=MIN_ENTROPY_BITS, min_skin_frac=MIN_SKIN_FRAC,
            skip_quality_checks=skip_quality,
        )
        if not face: continue

        # pHash de-dupe
        try:
            h = _phash64(face)
            if any(_hamming64(h, old) <= PHASH_HAMMING_NEAR for old in seen_hashes):
                continue
            seen_hashes.add(h)
        except Exception:
            pass

        if canon: seen_urls.add(canon)
        imgs.append(face); meta.append((murl, purl or murl, i))
        if len(imgs) >= need: break
    return imgs, meta

def collect_face_crops(query: str, need: int, *, first: str, last: str,
                       required_terms: Optional[List[str]] = None,
                       seen_urls: Optional[Set[str]] = None,
                       seen_hashes: Optional[Set[int]] = None) -> Tuple[List[Image.Image], List[Tuple[str,str,int]]]:
    if seen_urls is None: seen_urls = set()
    if seen_hashes is None: seen_hashes = set()
    tiles = bing_images_tiles_across_pages(query, want=max(need * 8, 60), max_pages=MAX_PAGES)

    imgs: List[Image.Image] = []
    meta: List[Tuple[str,str,int]] = []

    # Step 0: strict name, law/clerk if provided
    got, m = _collect_with_step(tiles, need, first=first, last=last, name_mode="strict",
                                required_terms=required_terms, seen_urls=seen_urls,
                                seen_hashes=seen_hashes, relax_level=0)
    imgs.extend(got); meta.extend(m)
    if len(imgs) >= need: return imgs, meta

    # Step 1: relaxed name (first + any last token), still enforce law/clerk on "Other"
    got, m = _collect_with_step(tiles, need-len(imgs), first=first, last=last, name_mode="relaxed",
                                required_terms=required_terms, seen_urls=seen_urls,
                                seen_hashes=seen_hashes, relax_level=1)
    imgs.extend(got); meta.extend(m)
    if len(imgs) >= need: return imgs, meta

    # Step 2: drop law/clerk requirement; loosen further (skip quality on high-trust)
    got, m = _collect_with_step(tiles, need-len(imgs), first=first, last=last, name_mode="relaxed",
                                required_terms=None, seen_urls=seen_urls,
                                seen_hashes=seen_hashes, relax_level=2)
    imgs.extend(got); meta.extend(m)
    if len(imgs) >= need: return imgs, meta

    # Step 3: broaden to plain "<First Last>" query (no law/clerk), then try again quickly
    if len(imgs) < need:
        q2 = f'"{first} {last}"'
        tiles2 = bing_images_tiles_across_pages(q2, want=max(need * 6, 50), max_pages=MAX_PAGES)
        got, m = _collect_with_step(tiles2, need-len(imgs), first=first, last=last, name_mode="relaxed",
                                    required_terms=None, seen_urls=seen_urls,
                                    seen_hashes=seen_hashes, relax_level=2)
        imgs.extend(got); meta.extend(m)

    return imgs, meta
def salvage_original_tiles(
    queries: List[str],
    need: int,
    *,
    first: str,
    last: str,
    seen_urls: Set[str],
    seen_hashes_any: Set[int],
    max_bytes: int = 8_000_000,      # relaxed byte cap for salvage
) -> Tuple[List[bytes], List[Tuple[str, str, int]]]:
    """
    Last-resort: accept ORIGINAL images (not face-verified crops) using relaxed/loose name match.
    We dedupe by canonical URL and by pHash of the original frame to avoid repeats.
    """
    imgs_b: List[bytes] = []
    meta: List[Tuple[str, str, int]] = []

    # Build a single tile pool from all salvage queries (earlier queries have higher priority).
    tiles: List[Dict] = []
    for q in queries:
        tiles.extend(bing_images_tiles_across_pages(q, want=max(need * 6, 50), max_pages=MAX_PAGES))

    for i, t in enumerate(tiles, start=1):
        purl = t.get("purl", ""); murl = t.get("murl", "")
        # Try relaxed, then loose, to broaden acceptance.
        if not _tile_about_person(t, first, last, name_mode="relaxed"):
            if not _tile_about_person(t, first, last, name_mode="loose"):
                continue

        # Canonical URL de-dupe
        canon = "|".join([_canon_url(t.get("murl","")),
                          _canon_url(t.get("turl","")),
                          _canon_url(t.get("purl",""))]).strip("|")
        if canon and canon in seen_urls:
            continue

        # Download with a larger budget
        b = download_tile_image(t, max_bytes=max_bytes)
        if not b:
            continue

        # Skip obviously non-photo URLs
        if _looks_like_art_url(murl) or _looks_like_art_url(purl):
            continue

        # pHash of ORIGINAL (not crop) to avoid same frame repeats, even via different CDNs.
        h = _phash_from_bytes(b)
        if h is not None:
            # Stricter than face-crop dedupe to avoid near-identical originals
            if any(_hamming64(h, old) <= 6 for old in seen_hashes_any):
                continue
            seen_hashes_any.add(h)

        if canon:
            seen_urls.add(canon)

        imgs_b.append(b)
        meta.append((murl, purl or murl, i))
        if len(imgs_b) >= need:
            break

    return imgs_b, meta

# ──────────────────────────────────────────────────────────────────────────────
# School picker (robust simple heuristic)
# ──────────────────────────────────────────────────────────────────────────────

def pick_school_name(education: str) -> str:
    if not education: return ""
    parts = [p.strip() for p in re.split(r"[;|,/]+", education.strip()) if p.strip()]
    if not parts: return education.strip()
    lowers = [p.lower() for p in parts]
    for i, s in enumerate(lowers):
        if "law" in s and "school" in s: return parts[i]
    for i, s in enumerate(lowers):
        s2 = s.replace(" ", "")
        if "juris" in s or "j.d" in s2 or "jd" in s2: return parts[i]
    for kw in ("university","college","school"):
        for i, s in enumerate(lowers):
            if kw in s: return parts[i]
    return parts[0]

# ──────────────────────────────────────────────────────────────────────────────
# LinkedIn-match (short-circuit) helper
# ──────────────────────────────────────────────────────────────────────────────

def try_linkedin_image_exact(first: str, last: str, linkedin_url: str) -> Optional[Dict]:
    if not linkedin_url: return None
    query = f'"{first} {last}" linkedin'
    tiles = bing_images_tiles_across_pages(query, want=30, max_pages=2)
    want_norm = normalize_linkedin_url(linkedin_url)
    best = None; best_sim = 0.0
    for t in tiles:
        purl = t.get("purl") or ""
        if "linkedin.com" not in purl and "licdn.com" not in purl: continue
        sim = linkedin_similarity(purl, want_norm)
        if sim > best_sim: best_sim = sim; best = t
        if sim >= 0.99: break
    if best and best_sim >= 0.6:
        b = download_tile_image(best)
        if b:
            return {"img_bytes": b, "murl": best.get("murl",""), "purl": best.get("purl",""), "search_query": query}
    return None

# ──────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────────────────────

def save_image_bytes_as_jpeg(dst_path: str, img_bytes: bytes) -> None:
    try:
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        img.save(dst_path, format="JPEG", quality=92)
    except Exception:
        with open(dst_path, "wb") as f:
            f.write(img_bytes)

def save_pil_jpeg(dst_path: str, img: Image.Image) -> None:
    img.convert("RGB").save(dst_path, format="JPEG", quality=92)

# ──────────────────────────────────────────────────────────────────────────────
# Name/gender priors
# ──────────────────────────────────────────────────────────────────────────────

def name_gender_prior(first: str) -> Tuple[Optional[str], float]:
    if not _GENDER: return None, 0.0
    g = (_GENDER.get_gender((first or "").strip()) or "").lower()
    if g in ("male","female"): return g, 0.95
    if g in ("mostly_male","mostly_female"): return g.split("_",1)[1], 0.75
    return None, 0.0

# ──────────────────────────────────────────────────────────────────────────────
# Person flows implementing the rules (with fixes)
# ──────────────────────────────────────────────────────────────────────────────

def process_li_match_only(first: str, middle: str, last: str, suffix: str,
                          li_hit: Dict, out_root: str, full_name: str, judge: str) -> Tuple[List[Dict], Dict]:
    prefix = file_prefix(first, middle, last, suffix)
    person_dir = os.path.join(out_root, folder_name(first, middle, last, suffix))
    _ensure_dir(person_dir)
    path1 = os.path.join(person_dir, f"{prefix}1.jpg")
    pathX = os.path.join(person_dir, f"{prefix}X.jpg")
    save_image_bytes_as_jpeg(path1, li_hit["img_bytes"])
    shutil.copy2(path1, pathX)

    image_row = {
        "row_type": "IMAGE","row_index": "",
        "name": full_name, "judge": judge,
        "person_file": os.path.basename(path1), "path": path1,
        "url": li_hit.get("murl",""), "source_cat": "LinkedIn",
        "weight_src": SOURCE_WEIGHTS.get("LinkedIn",5),
        "rank": 1, "cluster": 0, "in_majority": 1,
        "embedding_model": "none_original", "sim_to_centroid": 1.0,
        "gender_pred": "", "score_final": 1.0, "final_choice": 1,
        "race_asian": "", "race_indian": "", "race_black": "", "race_white": "",
        "race_middle_eastern": "", "race_latino_hispanic": "",
        "search_query": li_hit.get("search_query",""),
        "strategy": "linkedin_images_match", "padded_to_seven": 0
    }

    summary_row = {
        "row_type": "SUMMARY","row_index": "",
        "name": full_name, "judge": judge,
        "person_total_images": 1, "person_kept_images": 1,
        "person_majority_cluster": 0, "person_majority_ratio": 1.0,
        "majority_confidence": 1.0,
        "expected_gender_from_name": name_gender_prior(first)[0] or "",
        "deepface_gender_consistency": "skipped" if not _HAS_DEEPFACE else "checked",
        "person_dominant_race": "",
        "agg_asian_pct": "", "agg_indian_pct": "", "agg_black_pct": "", "agg_white_pct": "",
        "agg_middle_eastern_pct": "", "agg_latino_hispanic_pct": "",
        "src_Gov": 0, "src_Edu": 0, "src_Martindale-Hubbell": 0, "src_LinkedIn": 1,
        "src_Org": 0, "src_Firm": 0, "src_Professional": 0, "src_News": 0, "src_Other": 0,
        "query_used": li_hit.get("search_query",""),
        "strategy": "linkedin_images_match", "padded_to_seven": 0
    }
    return [image_row], summary_row

def process_person(first: str, middle: str, last: str, suffix: str,
                   judge: str, education: str, linkedin_url: str,
                   out_root: str, full_name: str) -> Tuple[List[Dict], Dict]:
    """
    Main per-person pipeline:
    - Try LinkedIn short-circuit (exact match → 1 image).
    - Otherwise build a 7-image bundle:
        1) _1.jpg = best LinkedIn/name-validated ORIGINAL tile.
        2) _2.. = face crops from "<First Last>" "Law Clerk" with staged fallbacks.
        3) Optional school image if LI absent and school available.
        3b) NEW: salvage ORIGINAL frames (non–face-verified) before padding if still short.
        4) Pad with copies of _1 only if we remain short after salvage.
        5) Duplicate _1 to _X.
    """
    prefix = file_prefix(first, middle, last, suffix)

    # Reset limited title-fetch budget
    reset_title_budget()

    # A) Try strict LinkedIn match short-circuit
    if linkedin_url:
        try:
            li_hit = try_linkedin_image_exact(first, last, linkedin_url)
        except Exception:
            li_hit = None
        if li_hit:
            return process_li_match_only(first, middle, last, suffix, li_hit, out_root, full_name, judge)

    # B) Build 7-image bundle
    image_rows: List[Dict] = []

    # 1) _1.jpg from "<First Last> linkedin" (valid face + name gate; prefer LinkedIn)
    q_li = f'"{first} {last}" linkedin'
    li_first = first_valid_linkedin_tile(q_li, first, last)
    if not li_first:
        li_first = first_downloadable_tile_face_checked(q_li, first, last)
    if not li_first:
        q_li_fallback = f'"{first} {last}" "Law Clerk"'
        li_first = first_downloadable_tile_face_checked(q_li_fallback, first, last)
    if not li_first:
        raise RuntimeError("No downloadable face-valid image found for LinkedIn-first tile.")

    # Create the person folder ONLY AFTER we have a valid _1.jpg (prevents blank folders)
    person_dir = os.path.join(out_root, folder_name(first, middle, last, suffix))
    _ensure_dir(person_dir)

    path1 = os.path.join(person_dir, f"{prefix}1.jpg")
    save_image_bytes_as_jpeg(path1, li_first["img_bytes"])

    strat = ("li_present_no_match_li1_lawclerk6" if linkedin_url else "li_absent_li1_lawclerk5_school1")
    image_rows.append({
        "row_type": "IMAGE","row_index": "",
        "name": full_name, "judge": judge,
        "person_file": os.path.basename(path1), "path": path1,
        "url": li_first.get("murl",""),
        "source_cat": _domain_category(li_first.get("purl","") or li_first.get("murl","")),
        "weight_src": SOURCE_WEIGHTS.get(_domain_category(li_first.get("purl","") or li_first.get("murl","")), 1),
        "rank": 1, "cluster": 0, "in_majority": 1,
        "embedding_model": "none_original", "sim_to_centroid": 0.0,
        "gender_pred": "", "score_final": 0.0, "final_choice": 1,
        "race_asian": "", "race_indian": "", "race_black": "", "race_white": "",
        "race_middle_eastern": "", "race_latino_hispanic": "",
        "search_query": li_first.get("search_query", q_li),
        "strategy": strat, "padded_to_seven": 0,
    })

    # Seed de-dupe with _1.jpg
    seen_urls: Set[str] = set()
    seen_hashes: Set[int] = set()
    try:
        canon1 = "|".join([
            _canon_url(li_first.get("murl","")),
            _canon_url(li_first.get("turl","")),
            _canon_url(li_first.get("purl",""))
        ]).strip("|")
        if canon1:
            seen_urls.add(canon1)

        # Prefer a face-crop pHash; if crop fails, fall back to original-frame pHash
        _1_face = extract_real_face(
            li_first["img_bytes"],
            murl=li_first.get("murl",""),
            purl=li_first.get("purl","") or li_first.get("murl",""),
        )
        if _1_face:
            seen_hashes.add(_phash64(_1_face))
        else:
            h_any = _phash_from_bytes(li_first["img_bytes"])
            if h_any is not None:
                seen_hashes.add(h_any)
    except Exception:
        pass

    # 2) _2.. from "<First Last>" "Law Clerk" with staged fallbacks
    q_lc = f'"{first} {last}" "Law Clerk"'
    need_lc = 6
    use_school_last = False
    school = pick_school_name(education)
    if not linkedin_url and school:
        need_lc = 5
        use_school_last = True

    lc_imgs, lc_meta = collect_face_crops(
        q_lc, need=need_lc, first=first, last=last,
        required_terms=LAW_KEYWORDS[:2],  # ["law","clerk"]
        seen_urls=seen_urls, seen_hashes=seen_hashes,
    )

    # If still short and we have a school, try a school-targeted query for extras
    if len(lc_imgs) < need_lc and school:
        q_school_extra = f'"{first} {last}" "{school}"'
        extra_imgs, extra_meta = collect_face_crops(
            q_school_extra, need=need_lc - len(lc_imgs), first=first, last=last,
            required_terms=None, seen_urls=seen_urls, seen_hashes=seen_hashes
        )
        lc_imgs.extend(extra_imgs)
        lc_meta.extend(extra_meta)

    race_keys = ["asian","indian","black","white","middle eastern","latino hispanic"]
    agg: Dict[str, List[float]] = {k: [] for k in race_keys}

    idx = 2
    for img, (murl, purl, rank) in zip(lc_imgs, lc_meta):
        fname = f"{prefix}{idx}.jpg"
        fpath = os.path.join(person_dir, fname)
        save_pil_jpeg(fpath, img)

        g_pred: str = ""
        race: Dict[str, float] = {}
        if _HAS_DEEPFACE:
            try:
                arr = np.array(img)
                res = DeepFace.analyze(arr, actions=["gender","race"], enforce_detection=False, silent=True)
                g_str = (res or {}).get("gender","")
                g_pred = ("female" if str(g_str).lower().startswith("f")
                          else ("male" if str(g_str).lower().startswith("m") else ""))
                race = (res or {}).get("race") or {}
            except Exception:
                g_pred, race = "", {}

        for k in race_keys:
            if k in race:
                try:
                    agg[k].append(float(race[k]))
                except Exception:
                    pass

        image_rows.append({
            "row_type": "IMAGE","row_index": "",
            "name": full_name, "judge": judge,
            "person_file": fname, "path": fpath,
            "url": murl, "source_cat": _domain_category(purl or murl),
            "weight_src": SOURCE_WEIGHTS.get(_domain_category(purl or murl), 1),
            "rank": rank, "cluster": 0, "in_majority": 1,
            "embedding_model": "face_crop_largest", "sim_to_centroid": 0.0,
            "gender_pred": g_pred, "score_final": 0.0, "final_choice": 0,
            "race_asian": race.get("asian","") if race else "",
            "race_indian": race.get("indian","") if race else "",
            "race_black": race.get("black","") if race else "",
            "race_white": race.get("white","") if race else "",
            "race_middle_eastern": race.get("middle eastern","") if race else "",
            "race_latino_hispanic": race.get("latino hispanic","") if race else "",
            "search_query": q_lc, "strategy": strat, "padded_to_seven": 0,
        })
        idx += 1
        if idx > (KEEP_IMAGES_TARGET if not use_school_last else KEEP_IMAGES_TARGET - 1):
            break

    # 3) Optional _7 from school
    if use_school_last and idx <= KEEP_IMAGES_TARGET:
        q_school = f'"{first} {last}" "{school}"'
        sch_imgs, sch_meta = collect_face_crops(q_school, need=1, first=first, last=last)
        if sch_imgs:
            img = sch_imgs[0]
            murl, purl, rank = sch_meta[0]
            fname = f"{prefix}{KEEP_IMAGES_TARGET}.jpg"
            fpath = os.path.join(person_dir, fname)
            save_pil_jpeg(fpath, img)

            g_pred = ""
            race = {}
            if _HAS_DEEPFACE:
                try:
                    arr = np.array(img)
                    res = DeepFace.analyze(arr, actions=["gender","race"], enforce_detection=False, silent=True)
                    g_str = (res or {}).get("gender","")
                    g_pred = ("female" if str(g_str).lower().startswith("f")
                              else ("male" if str(g_str).lower().startswith("m") else ""))
                    race = (res or {}).get("race") or {}
                    for k in race_keys:
                        if k in race:
                            agg[k].append(float(race[k]))
                except Exception:
                    g_pred, race = "", {}

            image_rows.append({
                "row_type": "IMAGE","row_index": "",
                "name": full_name, "judge": judge,
                "person_file": fname, "path": fpath,
                "url": murl, "source_cat": _domain_category(purl or murl),
                "weight_src": SOURCE_WEIGHTS.get(_domain_category(purl or murl), 1),
                "rank": rank, "cluster": 0, "in_majority": 1,
                "embedding_model": "face_crop_largest", "sim_to_centroid": 0.0,
                "gender_pred": g_pred, "score_final": 0.0, "final_choice": 0,
                "race_asian": race.get("asian","") if race else "",
                "race_indian": race.get("indian","") if race else "",
                "race_black": race.get("black","") if race else "",
                "race_white": race.get("white","") if race else "",
                "race_middle_eastern": race.get("middle eastern","") if race else "",
                "race_latino_hispanic": race.get("latino hispanic","") if race else "",
                "search_query": q_school, "strategy": strat, "padded_to_seven": 0,
            })
            idx += 1

    # 3b) SALVAGE ORIGINAL TILES (non–face-verified) BEFORE padding, if still short
    if idx <= KEEP_IMAGES_TARGET:
        salvage_need = KEEP_IMAGES_TARGET - (idx - 1)
        salvage_queries = [
            f'"{first} {last}"',
            f'"{first} {last}" portrait',
            f'"{first} {last}" headshot',
            f'"{first} {last}" bio',
            f'"{first} {last}" staff',
            f'"{first} {last}" team',
        ]
        if school:
            salvage_queries.append(f'"{first} {last}" "{school}"')

        salv_imgs_b, salv_meta = salvage_original_tiles(
            salvage_queries, need=salvage_need,
            first=first, last=last,
            seen_urls=seen_urls, seen_hashes_any=seen_hashes,
            max_bytes=8_000_000,
        )

        for b, (murl, purl, rank) in zip(salv_imgs_b, salv_meta):
            fname = f"{prefix}{idx}.jpg"
            fpath = os.path.join(person_dir, fname)
            save_image_bytes_as_jpeg(fpath, b)  # ORIGINAL frame saved (unverified)
            image_rows.append({
                "row_type": "IMAGE","row_index": "",
                "name": full_name, "judge": judge,
                "person_file": fname, "path": fpath,
                "url": murl, "source_cat": _domain_category(purl or murl),
                "weight_src": SOURCE_WEIGHTS.get(_domain_category(purl or murl), 1),
                "rank": rank, "cluster": 0, "in_majority": 1,
                "embedding_model": "original_unverified",  # labeled clearly
                "sim_to_centroid": 0.0,
                "gender_pred": "", "score_final": 0.0, "final_choice": 0,
                "race_asian": "", "race_indian": "", "race_black": "", "race_white": "",
                "race_middle_eastern": "", "race_latino_hispanic": "",
                "search_query": " | ".join(salvage_queries),
                "strategy": strat, "padded_to_seven": 0,
            })
            idx += 1
            if idx > KEEP_IMAGES_TARGET:
                break

    # 4) If still <7 images, pad with copies of _1 to meet contract
    while idx <= KEEP_IMAGES_TARGET:
        src = os.path.join(person_dir, f"{prefix}1.jpg")
        dst = os.path.join(person_dir, f"{prefix}{idx}.jpg")
        shutil.copy2(src, dst)
        image_rows.append({
            "row_type": "IMAGE","row_index": "",
            "name": full_name, "judge": judge,
            "person_file": os.path.basename(dst), "path": dst,
            "url": image_rows[0]["url"], "source_cat": image_rows[0]["source_cat"],
            "weight_src": image_rows[0]["weight_src"],
            "rank": 999, "cluster": 0, "in_majority": 1,
            "embedding_model": "duplicate_of_1", "sim_to_centroid": 1.0,
            "gender_pred": "", "score_final": 0.0, "final_choice": 0,
            "race_asian": "", "race_indian": "", "race_black": "", "race_white": "",
            "race_middle_eastern": "", "race_latino_hispanic": "",
            "search_query": image_rows[0]["search_query"], "strategy": strat, "padded_to_seven": 1,
        })
        idx += 1

    # 5) Duplicate _1.jpg to _X.jpg
    x_path = os.path.join(person_dir, f"{prefix}X.jpg")
    shutil.copy2(os.path.join(person_dir, f"{prefix}1.jpg"), x_path)

    # SUMMARY
    # (DeepFace aggregation left blank unless you enabled it above)
    src_breakdown = {"Gov":0,"Edu":0,"Martindale-Hubbell":0,"LinkedIn":0,"Org":0,"Firm":0,"Professional":0,"News":0,"Other":0}
    for r in image_rows:
        c = r.get("source_cat","Other")
        if c in src_breakdown:
            src_breakdown[c] += 1
        else:
            src_breakdown["Other"] += 1

    summary_row = {
        "row_type": "SUMMARY","row_index": "",
        "name": full_name, "judge": judge,
        "person_total_images": len(image_rows), "person_kept_images": len(image_rows),
        "person_majority_cluster": 0, "person_majority_ratio": 1.0, "majority_confidence": 1.0,
        "expected_gender_from_name": (name_gender_prior(first)[0] or ""),
        "deepface_gender_consistency": ("checked" if _HAS_DEEPFACE else "skipped"),
        "person_dominant_race": "",
        "agg_asian_pct": "", "agg_indian_pct": "", "agg_black_pct": "", "agg_white_pct": "",
        "agg_middle_eastern_pct": "", "agg_latino_hispanic_pct": "",
        "src_Gov": src_breakdown["Gov"], "src_Edu": src_breakdown["Edu"], "src_Martindale-Hubbell": src_breakdown["Martindale-Hubbell"],
        "src_LinkedIn": src_breakdown["LinkedIn"], "src_Org": src_breakdown["Org"], "src_Firm": src_breakdown["Firm"],
        "src_Professional": src_breakdown["Professional"], "src_News": src_breakdown["News"], "src_Other": src_breakdown["Other"],
        "query_used": q_lc, "strategy": strat,
        "padded_to_seven": 1 if any(r.get("padded_to_seven")==1 for r in image_rows) else 0,
    }

    # Ensure _1.._7 order (keep X last)
    def _order_key(r: Dict) -> int:
        fn = str(r.get("person_file",""))
        base = fn.split("_")[-1].split(".")[0] if fn else "999"
        return 1 if base == "X" else (int(base) if base.isdigit() else 999)

    image_rows = sorted(image_rows, key=_order_key)
    return image_rows, summary_row

# ──────────────────────────────────────────────────────────────────────────────
# I/O – column mapping & batch runner
# ──────────────────────────────────────────────────────────────────────────────

def map_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    cols = list(df.columns)
    norm = {c.lower().strip(): c for c in cols}
    def find(cands: List[str]) -> Optional[str]:
        for cand in cands:
            if cand in norm: return norm[cand]
        for c in cols:
            lc = c.lower().strip()
            if any(tok in lc for tok in cands): return c
        return None
    col_first = find(POSSIBLE_FIRST)
    col_middle = find(POSSIBLE_MIDDLE)
    col_last = find(POSSIBLE_LAST)
    col_suffix = find(POSSIBLE_SUFFIX)
    col_linked = find(POSSIBLE_LINKEDIN)
    col_edu = find(POSSIBLE_EDU)
    col_judge = find(POSSIBLE_JUDGE)
    if not col_first or not col_last:
        raise RuntimeError(f"Missing required First/Last columns. Headers seen: {', '.join(cols)}")
    return {
        "first": col_first, "middle": col_middle, "last": col_last, "suffix": col_suffix,
        "linkedin": col_linked, "education": col_edu, "judge": col_judge
    }

def get_row_value(row: pd.Series, col: Optional[str]) -> str:
    return _safe(row.get(col)) if col else ""

def full_name_from_row(row: pd.Series, cols: Dict[str, Optional[str]]) -> Tuple[str,str,str,str,str]:
    first = get_row_value(row, cols["first"])
    middle = get_row_value(row, cols["middle"])
    last = get_row_value(row, cols["last"])
    suffix = get_row_value(row, cols["suffix"])
    name = " ".join([p for p in [first, middle, last, suffix] if p])
    return name, first, middle, last, suffix

def run_batch(path: str) -> None:
    out_root = "batch_outputs"
    _ensure_dir(out_root)

    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    df = df.reset_index(drop=True)

    cols = map_columns(df)
    merged_csv = os.path.join(out_root, "images_and_summary.csv")

    race_cols = [f"race_{k}" for k in ["asian","indian","black","white","middle_eastern","latino_hispanic"]]
    image_fields = [
        "row_type","row_index","name","judge","person_file","path","url","source_cat","weight_src","rank",
        "cluster","in_majority","embedding_model","sim_to_centroid","gender_pred","score_final","final_choice",
        *race_cols,"search_query","strategy","padded_to_seven"
    ]
    summary_fields = [
        "row_type","row_index","name","judge","person_total_images","person_kept_images",
        "person_majority_cluster","person_majority_ratio","majority_confidence",
        "expected_gender_from_name","deepface_gender_consistency","person_dominant_race",
        "agg_asian_pct","agg_indian_pct","agg_black_pct","agg_white_pct","agg_middle_eastern_pct","agg_latino_hispanic_pct",
        "src_Gov","src_Edu","src_Martindale-Hubbell","src_LinkedIn","src_Org","src_Firm","src_Professional","src_News","src_Other",
        "query_used","strategy","padded_to_seven"
    ]
    union_fields = image_fields + [c for c in summary_fields if c not in image_fields]

    with open(merged_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=union_fields)
        writer.writeheader()
        total_rows = len(df)
        print(f"[INFO] Processing {total_rows} rows from {path} ...")
        for ridx, (_, row) in enumerate(df.iterrows(), start=1):
            name, first, middle, last, suffix = full_name_from_row(row, cols)
            if not first or not last:
                print(f"[SKIP] Row {ridx}: missing first/last.")
                continue
            judge = get_row_value(row, cols.get("judge"))
            edu = get_row_value(row, cols.get("education"))
            linkedin_url = get_row_value(row, cols.get("linkedin"))
            try:
                imgs, summ = process_person(first, middle, last, suffix, judge, edu, linkedin_url, out_root, name)
                for r in imgs:
                    r["row_index"] = ridx
                    writer.writerow({k: r.get(k, "") for k in union_fields})
                summ["row_index"] = ridx
                writer.writerow({k: summ.get(k, "") for k in union_fields})
                print(f"[OK] Row {ridx}/{total_rows}: {name} → {len(imgs)} image rows ({summ['strategy']})")
            except KeyboardInterrupt:
                print("\n[ABORT] Interrupted by user."); raise
            except Exception as e:
                print(f"[ERROR] Row {ridx}: {name} → {e}")
                err = {
                    "row_type": "ERROR","row_index": ridx,"name": name,"judge": judge,
                    "person_file": "","path": "","url": str(e),
                    "source_cat": "","weight_src": "","rank": "",
                    "cluster": "","in_majority": "","embedding_model": "","sim_to_centroid": "",
                    "gender_pred": "","score_final": "","final_choice": "",
                    "race_asian": "","race_indian": "","race_black": "","race_white": "",
                    "race_middle_eastern": "","race_latino_hispanic": "",
                    "search_query": "","strategy": "error","padded_to_seven": 0
                }
                writer.writerow({k: err.get(k, "") for k in union_fields})

    print(f"[OK] Wrote: {merged_csv}")

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) != 2 or os.path.splitext(sys.argv[1])[1].lower() not in {".csv",".xlsx",".xls"}:
        print("Usage:\n python main.py '<excel_or_csv>'")
        sys.exit(2)
    run_batch(sys.argv[1])
