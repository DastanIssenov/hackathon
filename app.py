
import streamlit as st
import pandas as pd
from datetime import datetime
import re
import urllib.request
import urllib.parse
import ssl

st.set_page_config(page_title="Link → CSV", layout="centered")

st.title("🔗 ➜ 📄 Link to CSV")
st.write("Paste a post or video URL. I'll grab useful metadata and give you a CSV to download.")

# --- URL input
url = st.text_input("Paste the full URL", placeholder="https://...")

# --- Options
with st.expander("Options"):
    inc_raw_html = st.checkbox("Include raw HTML (truncated)", value=False)
    timeout = st.number_input("Timeout (seconds)", min_value=5, max_value=60, value=20)

# --- Helpers: fetch and parse without extra dependencies
def fetch(url: str, timeout: int = 20) -> str:
    # user-agent to look like a browser
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        },
        method="GET",
    )
    context = ssl.create_default_context()
    with urllib.request.urlopen(req, timeout=timeout, context=context) as resp:
        charset = resp.headers.get_content_charset() or "utf-8"
        html = resp.read().decode(charset, errors="replace")
        final_url = resp.geturl()
    return html, final_url

# very small regex-based meta extractor (no bs4 needed)
_META_TAG_RE = re.compile(
    r'<meta\b[^>]*?(?:name|property)\s*=\s*["\\\']([^"\\\']+)["\\\'][^>]*?\scontent\s*=\s*["\\\']([^"\\\']*)["\\\'][^>]*?>',
    re.IGNORECASE | re.DOTALL,
)
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)

def extract_meta(html: str):
    meta = {}
    for m in _META_TAG_RE.finditer(html):
        key = m.group(1).strip()
        val = m.group(2).strip()
        meta[key] = val
    # title
    t = _TITLE_RE.search(html)
    if t and "title" not in meta and "og:title" not in meta:
        meta["title"] = re.sub(r"\\s+", " ", t.group(1)).strip()
    # derive description if none
    if "description" not in meta and "og:description" not in meta:
        # rough extract of first 160 chars of visible text
        txt = re.sub(r"<script.*?</script>|<style.*?</style>", " ", html, flags=re.DOTALL|re.IGNORECASE)
        txt = re.sub(r"<[^>]+>", " ", txt)
        txt = re.sub(r"\\s+", " ", txt).strip()
        if txt:
            meta["description_guess"] = txt[:200]
    return meta

def domain_of(u: str) -> str:
    try:
        return urllib.parse.urlparse(u).netloc or "unknown-domain"
    except Exception:
        return "unknown-domain"

# --- Main action
if st.button("Process link", type="primary", disabled=(not url)):
    try:
        with st.spinner("Fetching page..."):
            html, final_url = fetch(url, timeout=int(timeout))
        meta = extract_meta(html)

        # Common fields to surface
        row = {
            "input_url": url,
            "final_url": final_url,
            "domain": domain_of(final_url),
            "title": meta.get("og:title") or meta.get("twitter:title") or meta.get("title"),
            "description": meta.get("og:description") or meta.get("twitter:description") or meta.get("description") or meta.get("description_guess"),
            "image": meta.get("og:image") or meta.get("twitter:image"),
            "site_name": meta.get("og:site_name"),
            "published_time": meta.get("article:published_time") or meta.get("og:updated_time"),
            "author": meta.get("author") or meta.get("article:author"),
            "video_url": meta.get("og:video") or meta.get("twitter:player"),
        }

        # Include all meta as flattened columns (namespaced)
        for k, v in sorted(meta.items()):
            if k in row and row[k]:
                continue
            # prefix uncommon keys to avoid collisions
            if k not in ("og:title","twitter:title","title","og:description","twitter:description","description","description_guess","og:image","twitter:image","og:site_name","article:published_time","og:updated_time","author","article:author","og:video","twitter:player"):
                row[f"meta:{k}"] = v

        df = pd.DataFrame([row])

        st.success("Done! Preview below:")
        st.dataframe(df, use_container_width=True)

        # CSV download
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        fname = f"link_metadata_{domain_of(final_url)}_{ts}.csv"
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv_bytes, file_name=fname, mime="text/csv")

        # optional raw html (truncated)
        if inc_raw_html:
            st.subheader("Raw HTML (truncated)")
            st.code(html[:12000] + ("\\n... [truncated]" if len(html) > 12000 else ""), language="html")

    except Exception as e:
        st.error(f"Failed to process the link: {e}")
        st.exception(e)
else:
    st.info("Enter a URL and click **Process link**.")
