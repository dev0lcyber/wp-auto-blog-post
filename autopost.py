import sys
import os
import re
import requests
import random
from datetime import datetime, timedelta
import urllib.parse
import argparse
import base64
import logging
from groq import Groq
from PIL import Image
import json
import csv

# ==============================
# LOGGING & STARTUP
# ==============================
logging.basicConfig(
    filename='autopost.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
print("Kicking off the AI blog post generator! 🚀")

# ==============================
# CREDENTIALS (SET IN .env or BELOW)
# ==============================
# SECURITY: Never commit real keys! Use .env file or environment variables.
# Example .env:
#   GROQ_API_KEY=gsk_...
#   GROQ_API_KEY2=gsk_...
#   OPENROUTER_API_KEY=sk-or-v1-...
#   WP_SITE_URL=https://yoursite.com/
#   WP_USERNAME=your_username
#   WP_APP_PASSWORD=abcd efgh ijkl mnop qrst uvwx

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your-groq-key-1-here")
GROQ_API_KEY2 = os.getenv("GROQ_API_KEY2", "your-groq-key-2-here")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "your-openrouter-key-here")
WP_SITE_URL = os.getenv("WP_SITE_URL", "https://yourblog.com/").rstrip("/")
WP_USERNAME = os.getenv("WP_USERNAME", "your-wp-username")
WP_APP_PASSWORD = os.getenv("WP_APP_PASSWORD", "your-app-password-here")  # 24 chars, space-separated

# Optional: DeepAI fallback (not used yet)
DEEPAI_API_KEY = os.getenv("DEEPAI_API_KEY", "your-deepai-key")

# Validate required credentials
required = {
    "GROQ_API_KEY": GROQ_API_KEY,
    "GROQ_API_KEY2": GROQ_API_KEY2,
    "OPENROUTER_API_KEY": OPENROUTER_API_KEY,
    "WP_SITE_URL": WP_SITE_URL,
    "WP_USERNAME": WP_USERNAME,
    "WP_APP_PASSWORD": WP_APP_PASSWORD
}

missing = [k for k, v in required.items() if "your-" in v or not v.strip()]
if missing:
    print("ERROR: Missing or placeholder credentials:")
    for m in missing:
        print(f"   → {m}")
    print("\nSet them in a .env file or export as environment variables.")
    sys.exit(1)

# ==============================
# STATS FILE
# ==============================
STATS_FILE = "published_articles.csv"
if not os.path.exists(STATS_FILE):
    with open(STATS_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'Date', 'Link', 'Title', 'Categories'])

# ==============================
# ARGUMENTS
# ==============================
parser = argparse.ArgumentParser(description="Generate and optionally post AI/ML blog content to WordPress")
parser.add_argument("title", nargs='?', help="Article title")
parser.add_argument("--titles-file", help="Text file with titles (separated by '.')")
parser.add_argument("--post-to-wp", action="store_true", help="Auto-publish to WordPress")
args = parser.parse_args()

if not args.title and not args.titles_file:
    parser.error("Provide a title or use --titles-file.")

# Load titles
if args.titles_file:
    try:
        with open(args.titles_file, 'r', encoding='utf-8') as f:
            text = f.read()
        titles = [t.strip() for t in text.split('.') if t.strip()]
        if not titles:
            raise ValueError("No valid titles found.")
        print(f"Processing {len(titles)} titles from {args.titles_file}")
        logging.info(f"Processing {len(titles)} titles from {args.titles_file}")
    except Exception as e:
        logging.error(f"Error reading titles file: {e}")
        print(f"Error: {e}")
        sys.exit(1)
else:
    titles = [args.title]

# ==============================
# SETTINGS
# ==============================
base_output_dir = "blogs"
os.makedirs(base_output_dir, exist_ok=True)

# Category hierarchy: parent → child
category_hierarchy = {
    "AI History": "AI Fundamentals",
    "Beginner Resources": "AI Fundamentals",
    "Introduction to AI": "AI Fundamentals",
    "Key Concepts": "AI Fundamentals",
    "Emerging Tech": "AI Trends & Insights",
    "Future Outlook": "AI Trends & Insights",
    "Latest News": "AI Trends & Insights",
    "Bias & Fairness": "Ethical AI",
    "Case Studies": "Ethical AI",
    "Regulations": "Ethical AI",
    "Societal Impact": "Ethical AI",
    "Advanced Techniques": "ML Tutorials",
    "Beginner Projects": "ML Tutorials",
    "Flutter AI Apps": "ML Tutorials",
    "Python for ML": "ML Tutorials"
}

# ==============================
# TEXT API SELECTION WITH FAILOVER
# ==============================
current_text_api = None

def select_text_api():
    global current_text_api
    try:
        Groq(api_key=GROQ_API_KEY).chat.completions.create(
            messages=[{"role": "user", "content": "test"}], model="llama-3.3-70b-versatile", max_tokens=1)
        current_text_api = 0
        logging.info("Using Groq API #1")
        return
    except Exception as e:
        logging.warning(f"Groq #1 failed: {e}")
    try:
        Groq(api_key=GROQ_API_KEY2).chat.completions.create(
            messages=[{"role": "user", "content": "test"}], model="llama-3.3-70b-versatile", max_tokens=1)
        current_text_api = 1
        logging.info("Using Groq API #2")
        return
    except Exception as e:
        logging.warning(f"Groq #2 failed: {e}")
    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
            json={"model": "mistralai/mistral-7b-instruct:free", "messages": [{"role": "user", "content": "test"}]},
            timeout=10
        )
        resp.raise_for_status()
        current_text_api = 2
        logging.info("Using OpenRouter (free)")
        return
    except Exception as e:
        logging.error(f"All text APIs failed: {e}")
        raise Exception("All text generation APIs are down.")

select_text_api()

def call_llm(prompt, model="llama-3.3-70b-versatile", max_tokens=6000):
    global current_text_api
    while True:
        if current_text_api == 0:
            try:
                client = Groq(api_key=GROQ_API_KEY)
                resp = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=model, temperature=0.7, max_tokens=max_tokens
                )
                return resp.choices[0].message.content
            except Exception as e:
                logging.warning(f"Groq #1 failed: {e}. Falling back to Groq #2.")
                current_text_api = 1
        elif current_text_api == 1:
            try:
                client = Groq(api_key=GROQ_API_KEY2)
                resp = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=model, temperature=0.7, max_tokens=max_tokens
                )
                return resp.choices[0].message.content
            except Exception as e:
                logging.warning(f"Groq #2 failed: {e}. Falling back to OpenRouter.")
                current_text_api = 2
        else:
            try:
                resp = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "HTTP-Referer": "https://github.com/yourusername/wp-auto-blog-post",
                        "X-Title": "AI Blog Auto-Publisher"
                    },
                    json={
                        "model": "mistralai/mistral-7b-instruct:free",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens
                    },
                    timeout=60
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except Exception as e:
                logging.error(f"OpenRouter failed: {e}")
                raise

# ==============================
# UTILITIES
# ==============================
def random_past_datetime(days_back=30):
    """Return random datetime in the last N days."""
    days = random.randint(0, days_back - 1)
    time = timedelta(
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
        seconds=random.randint(0, 59)
    )
    return datetime.now() - timedelta(days=days) - time

# ==============================
# MAIN LOOP
# ==============================
for article_topic in titles:
    print(f"\nGenerating: '{article_topic}'")

    # Truncate long titles
    if len(article_topic) > 60:
        article_topic = article_topic[:60].rsplit(' ', 1)[0]
        logging.warning(f"Title truncated: {article_topic}")

    safe_slug = re.sub(r'[^a-z0-9-]', '', article_topic.lower().replace(' ', '-'))[:75]
    article_dir = os.path.join(base_output_dir, safe_slug)
    os.makedirs(article_dir, exist_ok=True)

    tmp_jpg_path = os.path.join(article_dir, f"{safe_slug}.jpg")
    final_webp_path = os.path.join(article_dir, f"{safe_slug}.webp")

    # -----------------------------
    # 1. GENERATE ARTICLE CONTENT
    # -----------------------------
    content_prompt = f"""
You are an expert AI/ML educator writing for developers and learners. Create a **2000–2500 word tutorial** on:

**Topic**: {article_topic}

**Requirements**:
- Output: **Clean HTML only** (`<h1>`, `<h2>`, `<pre><code>`, etc.)
- Primary keyword: `{article_topic}` (use in title, intro, one H2, one FAQ)
- Meta: Title (≤60 chars), Description (≤160), Keywords (5–10)
- Structure:
  1. `<!-- META: ... -->` comment at top
  2. `<h1>` with benefit (e.g., "Master {article_topic} in 2025")
  3. Intro, Prerequisites, Why It Matters, Key Benefits
  4. Step-by-step HOWTO (8–12 steps) with `<pre><code class="language-python">`
  5. Troubleshooting, Expert Tips, Case Study, Conclusion, FAQ
  6. FAQ with **valid JSON-LD schema** in `<script type="application/ld+json">`

**Style**:
- Friendly, expert tone
- Use emojis sparingly
- Include code, lists, tables, blockquotes
- No fake stats — cite "industry report 2024–2025" if needed

Write the full article now.
"""

    try:
        article_content = call_llm(content_prompt, max_tokens=6000)
        logging.info(f"Article generated: {article_topic}")
        print("Article generated!")
    except Exception as e:
        logging.error(f"Failed to generate article: {e}")
        print(f"Error: {e}")
        continue

    # -----------------------------
    # 2. EXTRACT CATEGORIES
    # -----------------------------
    cat_prompt = f"""
Given title: '{article_topic}', return 1–3 **exact** categories from this list (comma-separated):

AI Fundamentals, AI History, Beginner Resources, Introduction to AI, Key Concepts,
AI Trends & Insights, Emerging Tech, Future Outlook, Latest News,
Ethical AI, Bias & Fairness, Case Studies, Regulations, Societal Impact,
ML Tutorials, Advanced Techniques, Beginner Projects, Flutter AI Apps, Python for ML

Return only: Category One, Category Two
"""

    try:
        selected = call_llm(cat_prompt, max_tokens=100).strip()
        categories = [c.strip() for c in selected.split(",") if c.strip() in category_hierarchy.values() or c.strip() in category_hierarchy]
        if not categories:
            categories = ["ML Tutorials"]
        print(f"Categories: {', '.join(categories)}")
    except:
        categories = ["ML Tutorials"]

    # -----------------------------
    # 3. EXTRACT SEO METADATA
    # -----------------------------
    seo_prompt = f"""
For topic '{article_topic}':
- Meta Title (50–60 chars)
- Meta Description (150–160 chars)
- Keywords (5–10, comma-separated)

Format:
Meta Title: ...
Meta Description: ...
Keywords: ...
"""

    try:
        seo = call_llm(seo_prompt, max_tokens=300)
        meta_title = re.search(r'Meta Title:\s*(.+)', seo, re.I).group(1).strip()[:60]
        meta_desc = re.search(r'Meta Description:\s*(.+?)\s*Keywords:', seo, re.DOTALL).group(1).strip()[:160]
        keywords = re.search(r'Keywords:\s*(.+)', seo, re.I).group(1).strip()
    except:
        meta_title = f"{article_topic} - 2025 Guide"[:60]
        meta_desc = f"Learn {article_topic.lower()} with code, examples, and expert tips."[:160]
        keywords = f"{article_topic.lower()}, ai tutorial, machine learning"

    slug = safe_slug

    # -----------------------------
    # 4. FORMAT & CLEAN HTML
    # -----------------------------
    def clean_html(content):
        lines = [l for l in content.split('\n') if l.strip()]
        result = []
        in_table = False
        for line in lines:
            line = line.strip()
            if not line or any(x in line.lower() for x in ["image:", "prompt:", "featured image"]):
                continue
            if re.match(r'^(What|How|Why|When|Where|Who) .+\?$', line):
                result.append(f"<h2>{line}</h2>")
            elif re.match(r'^[A-Z][A-Za-z\s]+$', line) and len(line.split()) > 1:
                result.append(f"<h3>{line}</h3>")
            elif line.startswith('|') and line.endswith('|'):
                cells = [c.strip() for c in line.strip('|').split('|')]
                tag = "th" if any(h in cells[0].lower() for h in ["feature", "step", "benefit"]) else "td"
                row = "<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>"
                if not in_table:
                    result.append("<table style='border-collapse: collapse; width: 100%;'>")
                    in_table = True
                result.append(row)
            else:
                if in_table:
                    result.append("</table>")
                    in_table = False
                result.append(f"<p>{line}</p>")
        if in_table:
            result.append("</table>")
        return "\n".join(result)

    article_content = clean_html(article_content)

    # -----------------------------
    # 5. GENERATE WEBP IMAGE
    # -----------------------------
    image_prompt = f"modern sci-fi tech scene that visually represents {article_topic}."

    def generate_webp_image():
        try:
            print("Generating image via Pollinations.ai...")
            params = {
                "width": 1920, "height": 1080,
                "nologo": "True", "seed": random.randint(1, 999999999),
                "model": "flux"
            }
            url = f"https://image.pollinations.ai/prompt/{urllib.parse.quote(image_prompt)}"
            r = requests.get(url, params=params, timeout=200)
            r.raise_for_status()

            with open(tmp_jpg_path, "wb") as f:
                f.write(r.content)

            img = Image.open(tmp_jpg_path)
            img.save(final_webp_path, "WEBP", quality=80, method=6)
            os.remove(tmp_jpg_path)
            logging.info(f"WebP saved: {final_webp_path}")
            return final_webp_path
        except Exception as e:
            logging.error(f"Image failed: {e}")
            if os.path.exists(tmp_jpg_path):
                os.remove(tmp_jpg_path)
            return None

    image_path = generate_webp_image()
    image_filename = os.path.basename(image_path) if image_path else ""

    # -----------------------------
    # 6. SAVE TEXT FILE
    # -----------------------------
    txt_path = os.path.join(article_dir, f"{slug}_blog_post.txt")
    txt_content = f"""Article Content:
{article_content}

Categories:
{', '.join(categories)}

Main Image:
Filename: {image_filename}
Path: {image_path or "None"}
Prompt: {image_prompt}

SEO Metadata:
Meta Title: {meta_title}
Meta Description: {meta_desc}
Slug: {slug}
Keywords: {keywords}
"""
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(txt_content)
    print(f"Saved: {txt_path}")

    # -----------------------------
    # 7. PUBLISH TO WORDPRESS
    # -----------------------------
    if args.post_to_wp:
        print("Publishing to WordPress...")
        wp_json_url = f"{WP_SITE_URL}/wp-json/wp/v2"
        auth = base64.b64encode(f"{WP_USERNAME}:{WP_APP_PASSWORD}".encode()).decode()
        headers = {"Authorization": f"Basic {auth}", "Content-Type": "application/json"}

        try:
            requests.get(f"{wp_json_url}/posts", headers=headers, timeout=10).raise_for_status()
        except Exception as e:
            logging.error(f"WordPress API unreachable: {e}")
            continue

        def get_or_create_category(name):
            if not name: return None
            parent = category_hierarchy.get(name)
            parent_id = get_or_create_category(parent) if parent else None
            resp = requests.get(f"{wp_json_url}/categories?search={urllib.parse.quote(name)}", headers=headers)
            if resp.status_code == 200 and resp.json():
                for cat in resp.json():
                    if cat["name"] == name and (parent_id is None or cat["parent"] == parent_id):
                        return cat["id"]
            data = {"name": name, "slug": re.sub(r'[^a-z0-9-]', '', name.lower().replace(' ', '-'))}
            if parent_id: data["parent"] = parent_id
            r = requests.post(f"{wp_json_url}/categories", json=data, headers=headers)
            return r.json()["id"] if r.status_code == 201 else None

        def get_or_create_tags(tags_str):
            tags = [t.strip() for t in tags_str.split(",")[:10] if t.strip()]
            ids = []
            for tag in tags:
                r = requests.get(f"{wp_json_url}/tags?search={urllib.parse.quote(tag)}", headers=headers)
                if r.status_code == 200 and r.json():
                    ids.append(r.json()[0]["id"])
                else:
                    r = requests.post(f"{wp_json_url}/tags", json={"name": tag}, headers=headers)
                    if r.status_code == 201:
                        ids.append(r.json()["id"])
            return ids

        def upload_featured_image(img_path):
            if not img_path or not os.path.exists(img_path):
                return None
            filename = os.path.basename(img_path)
            try:
                with open(img_path, "rb") as f:
                    files = {"file": (filename, f, "image/webp")}
                    h = {"Authorization": headers["Authorization"]}
                    r = requests.post(f"{wp_json_url}/media", data={"alt_text": article_topic}, files=files, headers=h, timeout=30)
                    if r.status_code == 201:
                        media_id = r.json()["id"]
                        logging.info(f"Uploaded image ID: {media_id}")
                        return media_id
            except Exception as e:
                logging.error(f"Image upload failed: {e}")
            return None

        publish_dt = random_past_datetime(30)
        post_data = {
            "title": article_topic,
            "content": article_content,
            "excerpt": meta_desc,
            "slug": slug,
            "date": publish_dt.strftime('%Y-%m-%dT%H:%M:%S'),
            "status": "publish",
            "meta": {
                "rank_math_title": meta_title,
                "rank_math_description": meta_desc,
                "rank_math_focus_keyword": keywords.split(",")[0]
            }
        }
        post_data["categories"] = [get_or_create_category(c) for c in categories if get_or_create_category(c)]
        post_data["tags"] = get_or_create_tags(keywords)

        try:
            r = requests.post(f"{wp_json_url}/posts", json=post_data, headers=headers, timeout=15)
            if r.status_code == 201:
                post_id = r.json()["id"]
                post_link = r.json()["link"]
                print(f"Published: {post_link}")
                logging.info(f"Post ID {post_id}: {post_link}")

                if image_path:
                    media_id = upload_featured_image(image_path)
                    if media_id:
                        requests.post(f"{wp_json_url}/posts/{post_id}", json={"featured_media": media_id}, headers=headers)

                with open(STATS_FILE, 'a', newline='', encoding='utf-8') as f:
                    csv.writer(f).writerow([post_id, datetime.now().strftime('%Y-%m-%d %H:%M'), post_link, article_topic, ', '.join(categories)])
            else:
                logging.error(f"Post failed: {r.text}")
        except Exception as e:
            logging.error(f"WP post error: {e}")

print("\nAll done! Check blogs/ and WordPress dashboard.")
logging.info("Script completed successfully.")
