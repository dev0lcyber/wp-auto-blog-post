import sys
import os
import re
import requests
import random
import urllib.parse
import argparse
import base64
import logging
from groq import Groq
from PIL import Image
import json
import csv
from datetime import datetime

# Setup logging with current date and time
logging.basicConfig(filename='autopost.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
print("Kicking off the blog post generator with some eco-energy vibes! [Started at 09:39 PM +01, Oct 25, 2025]")

# Hardcoded credentials (ALL OF THEM ARE FREE VERSION)
GROQ_API_KEY = "INSERT YOU API KEY HERE "  # Primary
GROQ_API_KEY2 = "INSERT YOU API KEY HERE "  # Secondary
OPENROUTER_API_KEY = "INSERT YOU API KEY HERE "
WP_SITE_URL = "INSERT YOU WEBSITE URL HERE "
WP_USERNAME = "INSERT WORDPRESS USERNAME HERE"
WP_APP_PASSWORD = "INSERT YOUR WORDPRESS USER APPLICAITON PASSWORD HERE"
DEEPAI_API_KEY = "INSERT YOU API KEY HERE"

# Statistics file
STATS_FILE = "published_articles.csv"
if not os.path.exists(STATS_FILE):
    with open(STATS_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'Date', 'Link', 'Title', 'Categories'])

# Command-line argument parser
parser = argparse.ArgumentParser(description="Generate and optionally post blog content to WordPress with !")
parser.add_argument("title", nargs='?', help="Article title (e.g., 'Solar Energy'). Optional if --titles-file is used.")
parser.add_argument("--titles-file", help="Text file with multiple titles separated by dots (e.g., 'Title1. Title2. Title3.')")
parser.add_argument("--post-to-wp", action="store_true", help="Auto-post to WordPress like a boss!")
args = parser.parse_args()

if not args.title and not args.titles_file:
    parser.error("Either provide a title or use --titles-file.")

# Get titles
if args.titles_file:
    try:
        with open(args.titles_file, 'r', encoding='utf-8') as f:
            text = f.read()
        titles = [t.strip() for t in text.split('.') if t.strip()]
        if not titles:
            raise ValueError("No valid titles found in the file.")
        print(f"Processing {len(titles)} titles from file: {args.titles_file}")
        logging.info(f"Processing {len(titles)} titles from file: {args.titles_file}")
    except Exception as e:
        logging.error(f"Error reading titles file: {e}")
        print(f"Error reading titles file: {e}")
        sys.exit(1)
else:
    titles = [args.title]

# Define parameters
word_count_range = (2000, 2500)
base_output_dir = "blogs"

# Dynamic category selection
category_map = {
    # AI Fundamentals
    "fundamentals": ["AI Fundamentals"], "introduction": ["AI Fundamentals"], 
    "key concepts": ["AI Fundamentals"], "history": ["AI History"], 
    "beginner": ["Beginner Resources"], "basics": ["Introduction to AI"],
    
    # AI Trends & Insights  
    "trends": ["AI Trends & Insights"], "emerging": ["Emerging Tech"],
    "future": ["Future Outlook"], "latest": ["Latest News"],
    
    # Ethical AI
    "ethical": ["Ethical AI"], "bias": ["Bias & Fairness"],
    "case study": ["Case Studies"], "regulation": ["Regulations"],
    "societal": ["Societal Impact"],
    
    # ML Tutorials
    "tutorial": ["ML Tutorials"], "python": ["Python for ML"],
    "flutter": ["Flutter AI Apps"], "beginner project": ["Beginner Projects"],
    "advanced": ["Advanced Techniques"]
}

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

# Ensure base blogs directory exists
if not os.path.exists(base_output_dir):
    os.makedirs(base_output_dir)
    logging.info(f"Created base directory: {base_output_dir}")

# Global for text API selection
current_text_api = None  # 0: Groq1, 1: Groq2, 2: OpenRouter

def select_text_api():
    global current_text_api
    # Try Groq1
    try:
        client = Groq(api_key=GROQ_API_KEY)
        client.chat.completions.create(messages=[{"role":"user","content":"test"}], model="llama-3.3-70b-versatile", max_tokens=1)
        current_text_api = 0
        return
    except: pass
    # Try Groq2
    try:
        client = Groq(api_key=GROQ_API_KEY2)
        client.chat.completions.create(messages=[{"role":"user","content":"test"}], model="llama-3.3-70b-versatile", max_tokens=1)
        current_text_api = 1
        return
    except: pass
    # Fallback OpenRouter
    try:
        resp = requests.post("https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
            json={"model":"mistralai/mistral-7b-instruct:free","messages":[{"role":"user","content":"test"}]}
        )
        resp.raise_for_status()
        current_text_api = 2
        return
    except:
        raise Exception("All text APIs failed on startup")

# Call once after titles
select_text_api()

def call_llm(prompt, model="llama-3.3-70b-versatile", max_tokens=6000):
    global current_text_api
    while True:
        if current_text_api == 0:
            try:
                client = Groq(api_key=GROQ_API_KEY)
                resp = client.chat.completions.create(messages=[{"role":"user","content":prompt}], model=model, temperature=0.7, max_tokens=max_tokens)
                return resp.choices[0].message.content
            except Exception as e:
                logging.warning(f"Groq1 failed: {e}. Switching to Groq2.")
                current_text_api = 1
                continue
        elif current_text_api == 1:
            try:
                client = Groq(api_key=GROQ_API_KEY2)
                resp = client.chat.completions.create(messages=[{"role":"user","content":prompt}], model=model, temperature=0.7, max_tokens=max_tokens)
                return resp.choices[0].message.content
            except Exception as e:
                logging.warning(f"Groq2 failed: {e}. Switching to OpenRouter.")
                current_text_api = 2
                continue
        else:
            try:
                resp = requests.post("https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}","HTTP-Referer":"https://EXEMPLE.lovestoblog.com","X-Title":"EXEMPLE"},
                    json={"model":"mistralai/mistral-7b-instruct:free","messages":[{"role":"user","content":prompt}]}
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except Exception as e:
                raise Exception(f"OpenRouter failed: {e}")

for article_topic in titles:
    # Validate title length
    if len(article_topic) > 60:
        logging.warning(f"Title '{article_topic}' exceeds 60 characters, truncating to fit.")
        print(f"Title '{article_topic}' too long, truncating to 60 chars.")
        article_topic = article_topic[:60].rsplit(' ', 1)[0]

    print(f"Generating a stellar article on '{article_topic}'...")

    # Create per-article folder under blogs/
    safe_slug = re.sub(r'[^a-z0-9-]', '', article_topic.lower().replace(' ', '-'))[:75]
    article_dir = os.path.join(base_output_dir, safe_slug)
    if not os.path.exists(article_dir):
        os.makedirs(article_dir)
        logging.info(f"Created article directory: {article_dir}")
    image_path = os.path.join(article_dir, f"{safe_slug}.jpg")

    # Generate article content
    content_prompt = f"""
You are an expert AI/ML journalist and senior SEO editor writing for EXEMPLE.me (audience: global developers, beginners, professionals). Write helpful, trustworthy, high-quality tutorial content that's valuable for learning, with clear, concise explanations, practical examples, and educational depth. Use a friendly, confident tone to make complex topics accessible and engaging for users building skills.

REQUIREMENTS:
- LENGTH: 2000-2500 words (minimum 1,900 if needed). 
- OUTPUT: Clean HTML only (<h1>, <h2>, <h3>, <p>, <ul>, <ol>, <li>, <b>, <blockquote>, <table>, <img>, <q>, <pre>, <code>, <script>).
- SEO: Primary keyword = '{article_topic}'. Include 3–6 related LSI keywords from your dataset.
  Keyword density: ~1%. Use it in the title, intro, one <h2>, and one FAQ answer.
- META: Add meta title (≤60 chars), meta description (≤155 chars), and slug suggestion.
- DATA SAFETY: No fabricated stats. If data is uncertain, say “(source: recent industry report 2024–2025)”.
- CODE HANDLING: For tutorials (HOWTO format) or where relevant, include practical code snippets wrapped in <pre><code> ... </code></pre> tags. Use syntax highlighting-friendly formatting (e.g., indent code properly). Ensure code is accurate, testable, and explained in adjacent <p> tags.
- EDUCATIONAL FOCUS: Emphasize learning outcomes, prerequisites, step-by-step guidance, tips for common pitfalls, and real-world applications to provide maximum value for learners. Keep all explanations concise.
- SCHEMA HANDLING: For JSON-LD FAQ schema, wrap in <script type="application/ld+json"> ... </script> to ensure it's not visible on the page.

STRUCTURE:
1. HTML comment block at top: meta title, meta description, slug, and word count.
2. <h1> — Article title (use '{article_topic}'+ benefit or hook, e.g., 'Master {article_topic} for Practical AI Skills').
3. Intro (150–220 words): engaging hook with '{article_topic}' early, outline learning objectives.
4. <h2> Prerequisites — 100–150 words (list required knowledge/tools).
5. <h2> Why This Matters — 200–300 words (explain real-world value).
6. <h2> Key Benefits — bulleted list (120–180 words, include emojis, focus on learning gains).
7. <h2> Main Section — depends on format, prioritize HOWTO for tutorials:
   - LISTICLE: 8–12 numbered items, each with <h3>, 2–3 <p>, and one practical tip or warning.
   - HOWTO: 8–12 steps, each with <h3>, concise <p> explanations, code snippet in <pre><code> if applicable, visuals tips, and checklist (<ol>/<ul>) for key takeaways.
   - REVIEW: 5 products/services, each with <h3>, summary, pros, cons, specs, and score (out of 10).
8. <h2> Troubleshooting Common Issues — 200–300 words (list 5–8 common problems with concise solutions, use <ul> or <ol>).
9. <h2> Expert Tips — 150–200 words (advanced insights for deeper learning).
10. <h2> Case Study or Example — 150–200 words (mention global context, demonstrate application).
11. <h2> Conclusion — 120–180 words (summarize key learnings, suggest next steps).
12. <h2> FAQ — 3 questions with '{article_topic}' in at least one. Follow with valid JSON-LD FAQ schema wrapped in <script type="application/ld+json"> {json} </script>.

STYLE:
- Use <p> for paragraphs (2–3 sentences max, keep concise).
- Use <b> for emphasis.
- Include 4–8 bullet lists total.
- Add 1–2 emojis per major section (not in meta/title).
- Include at least one <blockquote> and one numbered checklist.
- For code: Wrap in <pre><code class="language-python"> ... </code></pre> (or appropriate language class) to enable syntax highlighting in WordPress.
- Output must be valid, clean HTML only — no inline styles, CSS, or Markdown.

Now write the article for:
TOPIC: {article_topic}
MAIN_KEYWORD: {article_topic}
FORMAT: HOWTO
TONE: friendly expert
"""


    try:
        article_content = call_llm(content_prompt, max_tokens=6000)
        logging.info(f"Article generated successfully for '{article_topic}'")
        print("Article generated! Ready to make it shine with some HTML magic...")
    except Exception as e:
        logging.error(f"Error generating article for '{article_topic}': {e}")
        print(f"Oops, something broke: Error generating article: {e}")
        continue

    # AI selects 1–3 categories from your menu
    cat_prompt = f"""
You are an expert content strategist. Given the article title '{article_topic}' and its content, 
select 1–3 **exact** categories from this menu (use full names, case-sensitive):

- AI Fundamentals
  - AI History
  - Beginner Resources
  - Introduction to AI
  - Key Concepts
- AI Trends & Insights
  - Emerging Tech
  - Future Outlook
  - Latest News
- Ethical AI
  - Bias & Fairness
  - Case Studies
  - Regulations
  - Societal Impact
- ML Tutorials
  - Advanced Techniques
  - Beginner Projects
  - Flutter AI Apps
  - Python for ML

Return only the selected categories as a comma-separated list. Example: AI Fundamentals, Python for ML
"""

    try:
        selected_cats = call_llm(cat_prompt, max_tokens=100).strip()
        categories = [c.strip() for c in selected_cats.split(",") if c.strip() in 
                      ["AI Fundamentals","AI History","Beginner Resources","Introduction to AI","Key Concepts",
                       "AI Trends & Insights","Emerging Tech","Future Outlook","Latest News",
                       "Ethical AI","Bias & Fairness","Case Studies","Regulations","Societal Impact",
                       "ML Tutorials","Advanced Techniques","Beginner Projects","Flutter AI Apps","Python for ML"]]
        if not categories:
            categories = ["AI Fundamentals"]
        print(f"AI selected categories: {', '.join(categories)}")
    except:
        categories = ["AI Fundamentals"]

    # Generate SEO metadata dynamically
    seo_prompt = f"""
    You are an SEO expert. Based on the article topic '{article_topic}', generate:
    - A unique, SEO-friendly meta title (50-60 characters, include primary keyword, make it compelling and varied, e.g., not always starting with 'Why'). Must not exceed 60 characters.
    - A meta description (150-160 characters, engaging, include keywords, call to action). Must not exceed 160 characters.
    - Focus keywords (comma-separated list of 5-10 keywords, including primary, variations, and long-tail).
    Output in plain text format:
    Meta Title: [title]
    Meta Description: [description]
    Keywords: [keywords]
    """

    try:
        seo_content = call_llm(seo_prompt, max_tokens=300)
        meta_title_match = re.search(r'Meta Title:\s*(.+)', seo_content)
        meta_description_match = re.search(r'Meta Description:\s*(.+?)\s*Keywords:', seo_content, re.DOTALL)
        keywords_match = re.search(r'Keywords:\s*(.+)', seo_content)
        
        meta_title = meta_title_match.group(1).strip()[:60] if meta_title_match else f"{article_topic}: Eco Benefits & Tips"[:60]
        meta_description = meta_description_match.group(1).strip()[:160] if meta_description_match else f"Explore {article_topic.lower()} for sustainable homes in USA, Canada, EU. Benefits, tips, eco guides."[:160]
        keywords = keywords_match.group(1).strip() if keywords_match else f"{article_topic.lower()}, sustainable living, eco-friendly technology"
        
        logging.info(f"SEO metadata generated for '{article_topic}'")
        print("SEO metadata generated!")
        logging.info(f"Meta title length: {len(meta_title)} chars, Meta description length: {len(meta_description)} chars")
    except Exception as e:
        logging.error(f"Error generating SEO metadata for '{article_topic}': {e}")
        print(f"Error generating SEO metadata: {e}")
        meta_title = f"{article_topic}: Eco Benefits & Tips"[:60]
        meta_description = f"Explore {article_topic.lower()} for sustainable homes in USA, Canada, EU. Benefits, tips, eco guides."[:160]
        keywords = f"{article_topic.lower()}, sustainable living, eco-friendly technology"
        logging.info(f"Fallback SEO metadata used for '{article_topic}'")

    # Generate slug (permalink) ≤75 characters
    slug = safe_slug
    logging.info(f"Generated slug: {slug} ({len(slug)} chars)")

    # Set post title to the original topic (for <h1> and post title)
    post_title = article_topic

    # Add HTML formatting (H1, H2, H3, tables, line breaks)
    def format_article_content(content, title):
        formatted_content = ""
        lines = content.split('\n')
        formatted_lines = []
        for line in lines:
            lower_line = line.lower()
            if (
                (len(line.split()) > 10 and "beautiful modern eco-friendly scene" in lower_line)
                or "image:" in lower_line
                or "caption:" in lower_line
                or "visual:" in lower_line
                or "illustration:" in lower_line
                or "prompt:" in lower_line
                or "scene that visually represents" in lower_line
                or "insert image" in lower_line
                or "featured image" in lower_line
            ):
                continue
            elif re.match(r'^(What|How|Why|When|Where|Who) .+\?$', line, re.IGNORECASE):
                formatted_lines.append(f"<h2>{line}</h2>")
            elif re.match(r'^[A-Z][a-zA-Z\s]+$', line) and len(line.split()) > 1:
                formatted_lines.append(f"<h3>{line}</h3>")
            else:
                line = re.sub(r'\.(\s+)', r'.<br>\n', line)
                formatted_lines.append(line)
        
        table_pattern = r'^\|(.+?)\|\s*$'
        table_lines = []
        in_table = False
        for line in formatted_lines:
            if re.match(table_pattern, line):
                if not in_table:
                    table_lines.append("<table border='1' style='border-collapse: collapse; width: 100%;'>")
                    in_table = True
                cells = [cell.strip() for cell in line.strip('|').split('|')]
                if cells and cells[0].lower().replace(' ', '') in ['benefit', 'feature', 'step', 'cost', 'item']:
                    table_lines.append("<tr>" + "".join(f"<th style='padding: 8px;'>{cell}</th>" for cell in cells) + "</tr>")
                else:
                    table_lines.append("<tr>" + "".join(f"<td style='padding: 8px;'>{cell}</td>" for cell in cells) + "</tr>")
            else:
                if in_table:
                    table_lines.append("</table>")
                    in_table = False
                table_lines.append(line)
        if in_table:
            table_lines.append("</table>")
        
        formatted_content += "\n".join(table_lines)
        return formatted_content

    article_content = format_article_content(article_content, post_title)

    # Generate image (Pollinations.ai → DeepAI)
    image_filename = f"{safe_slug}.jpg"
    image_prompt = f'''hand drawn illustration, blue dominant palette, tech style, 
    soft pencil shading, subtle drop shadows, developer coding {article_topic}, 
    laptop dual monitors glowing code diagrams, whiteboard AI flowchart sticky notes,
     cables router coffee mug notebooks, futuristic office or server room, detailed professional,
      no text, 1200x630'''
    def generate_image(prompt, width=1920, height=1080):
        # Pollinations.ai
        try:
            print("Generating image with Pollinations.ai...")
            params = {"width": width, "height": height, "nologo": True, "seed": random.randint(1, 999999999), "model": "flux"}
            encoded_prompt = urllib.parse.quote(prompt)
            url = f"https://image.pollinations.ai/prompt/{encoded_prompt}"
            response = requests.get(url, params=params, timeout=60)
            if response.status_code == 200:
                with open(image_path, "wb") as f:
                    f.write(response.content)
                img = Image.open(image_path)
                img_data = list(img.getdata())
                new_img = Image.new(img.mode, img.size)
                new_img.putdata(img_data)
                new_img.save(image_path, "JPEG", quality=95)
                logging.info(f"All metadata removed from {image_path}")
                print(f"All metadata removed from {image_path}")
                return True
        except Exception as e:
            logging.info(f"2nd Attempt to generate the image")
            
            params = {"width": width, "height": height, "nologo": True, "seed": random.randint(1, 999999999), "model": "flux"}
            encoded_prompt = urllib.parse.quote(prompt)
            url = f"https://image.pollinations.ai/prompt/{encoded_prompt}"
            response = requests.get(url, params=params, timeout=60)
            if response.status_code == 200:
                with open(image_path, "wb") as f:
                    f.write(response.content)
                img = Image.open(image_path)
                img_data = list(img.getdata())
                new_img = Image.new(img.mode, img.size)
                new_img.putdata(img_data)
                new_img.save(image_path, "JPEG", quality=95)
                logging.info(f"All metadata removed from {image_path}")
                print(f"All metadata removed from {image_path}")
                return True

                
        # DeepAI fallback
        try:
            print("Trying DeepAI as fallback...")
            response = requests.post(
                "https://api.deepai.org/api/text2img",
                data={"text": prompt, "grid_size": "1"},
                headers={"api-key": DEEPAI_API_KEY},
                timeout=15
            )
            if response.status_code == 200 and "output_url" in response.json():
                img_url = response.json()["output_url"]
                img_response = requests.get(img_url, timeout=15)
                if img_response.status_code == 200:
                    with open(image_path, "wb") as f:
                        f.write(img_response.content)
                    img = Image.open(image_path)
                    img_data = list(img.getdata())
                    new_img = Image.new(img.mode, img.size) 
                    new_img.putdata(img_data)
                    new_img.save(image_path, "JPEG", quality=75)
                    logging.info(f"All metadata removed from {image_path}")
                    print(f"All metadata removed from {image_path}")
                    return True
        except Exception as e:
            logging.error(f"DeepAI error: {e}")
            print(f"DeepAI error: {e}")
        return False

    image_result = generate_image(image_prompt)
    if image_result:
        # Convert to WebP
        img = Image.open(image_path)
        webp_path = image_path.rsplit('.', 1)[0] + '.webp'
        img.save(webp_path, 'WEBP', quality=70)
        logging.info(f"Converted to WebP: {webp_path}")
        print(f"Converted to WebP: {webp_path}")
        image_path = webp_path
        image_filename = os.path.basename(webp_path)
    else:
        print(f"Image generation failed for '{article_topic}'. Generate manually at https://pollinations.ai or https://deepai.org.")
        logging.warning(f"Image generation failed for '{article_topic}'")

    # Save to .txt inside article folder
    output_filename = os.path.join(article_dir, f"{slug}_blog_post.txt")
    txt_content = f"""Article Content:
{article_content}

Categories:
{', '.join(categories)}

Main Image:
Filename: {image_filename}
Path: {image_path}
Prompt: {image_prompt}

SEO Metadata:
Meta Title: {meta_title}
Meta Description: {meta_description}
Slug: {slug}
Keywords: {keywords}"""

    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(txt_content)
        logging.info(f"Blog post saved to {output_filename}")
        print(f"Blog post saved to {output_filename}")
    except Exception as e:
        logging.error(f"Error saving file for '{article_topic}': {e}")
        print(f"Error saving file: {e}")
        continue

    # Auto-post to WordPress
    if args.post_to_wp:
        print(f"Starting WordPress auto-post for '{article_topic}'... Let’s get this live!")
        logging.info(f"Starting WordPress auto-post for '{article_topic}'")
        
        wp_json_url = f"{WP_SITE_URL.rstrip('/')}/wp-json/wp/v2"
        auth_str = f"{WP_USERNAME}:{WP_APP_PASSWORD}"
        b64_auth = base64.b64encode(auth_str.encode()).decode()
        headers = {
            "Authorization": f"Basic {b64_auth}",
            "Content-Type": "application/json"
        }
        
        # Test API connectivity
        try:
            test_response = requests.get(f"{wp_json_url}/posts", headers=headers, timeout=10)
            if test_response.status_code != 200:
                error_msg = f"Error connecting to WordPress API: {test_response.status_code} - {test_response.text}"
                logging.error(error_msg)
                print(f"{error_msg}")
                continue
            logging.info("WordPress API connectivity test successful")
            print("WordPress API is ready to rock!")
        except Exception as e:
            error_msg = f"Error testing WordPress API: {e}"
            logging.error(error_msg)
            print(f"{error_msg}")
            continue
        
        def get_or_create_category(name):
            if name is None:
                return None
            parent_name = category_hierarchy.get(name, None)
            parent_id = get_or_create_category(parent_name) if parent_name else None
            all_cats = []
            page = 1
            while True:
                response = requests.get(f"{wp_json_url}/categories?per_page=100&page={page}", headers=headers, timeout=10)
                if response.status_code != 200:
                    break
                data = response.json()
                if not data:
                    break
                all_cats.extend(data)
                page += 1
            for cat in all_cats:
                if cat["name"].lower() == name.lower() and (parent_id is None or cat["parent"] == parent_id):
                    return cat["id"]
            create_data = {
                "name": name,
                "slug": re.sub(r'[^a-z0-9-]', '', name.lower().replace(' ', '-'))
            }
            if parent_id:
                create_data["parent"] = parent_id
            create_resp = requests.post(f"{wp_json_url}/categories", json=create_data, headers=headers, timeout=10)
            if create_resp.status_code == 201:
                logging.info(f"Created category: {name} with parent {parent_name}")
                print(f"Created category: {name} with parent {parent_name}")
                return create_resp.json()["id"]
            else:
                logging.error(f"Failed to create category {name}: {create_resp.text}")
                print(f"Failed to create category {name}: {create_resp.text}")
            return None
        
        def get_or_create_tags(tags_str):
            tags = [tag.strip() for tag in tags_str.split(",") if tag.strip()]
            tag_ids = []
            for tag in tags[:10]:
                try:
                    response = requests.get(f"{wp_json_url}/tags?search={urllib.parse.quote(tag)}", headers=headers, timeout=10)
                    if response.status_code == 200 and response.json():
                        tag_ids.append(response.json()[0]["id"])
                    else:
                        create_data = {"name": tag, "slug": re.sub(r'[^a-z0-9-]', '', tag.lower().replace(' ', '-'))}
                        create_resp = requests.post(f"{wp_json_url}/tags", json=create_data, headers=headers, timeout=10)
                        if create_resp.status_code == 201:
                            tag_ids.append(create_resp.json()["id"])
                            logging.info(f"Created tag: {tag}")
                            print(f"Created tag: {tag}")
                except:
                    pass
            return tag_ids
        
        def upload_featured_image(image_path):
            if not os.path.exists(image_path):
                return None
            try:
                with open(image_path, "rb") as img_file:
                    data = {"caption": "", "description": article_topic, "alt_text": ""}
                    mime_type = "image/webp" if image_path.endswith('.webp') else "image/jpeg"
                    files = {"file": (image_filename, img_file, mime_type)}
                    media_headers = {"Authorization": headers["Authorization"], "Content-Disposition": f'attachment; filename={image_filename}'}
                    response = requests.post(f"{wp_json_url}/media", data=data, files=files, headers=media_headers, timeout=15)
                    if response.status_code == 201:
                        media_id = response.json()["id"]
                        logging.info(f"Featured image uploaded: {media_id}")
                        print(f"Featured image uploaded: {media_id}")
                        return media_id
            except:
                pass
            return None
        
        # Create post
        post_data = {
            "title": article_topic,
            "content": article_content,
            "excerpt": meta_description,
            "slug": slug,
            "status": "publish",
            "featured_media": None,
            "meta": {
                "rank_math_title": meta_title,
                "rank_math_description": meta_description,
                "rank_math_focus_keyword": keywords
            }
        }
        
        category_ids = [get_or_create_category(cat) for cat in categories if cat]
        post_data["categories"] = [cat_id for cat_id in category_ids if cat_id]
        post_data["tags"] = get_or_create_tags(keywords)
        
        try:
            response = requests.post(f"{wp_json_url}/posts", json=post_data, headers=headers, timeout=10)
            if response.status_code == 201:
                post_id = response.json()["id"]
                post_link = response.json()["link"]
                success_msg = f"Post created successfully! ID: {post_id} | View: {WP_SITE_URL}/wp-admin/post.php?post={post_id}&action=edit"
                logging.info(success_msg)
                print(f"{success_msg}")
                featured_id = upload_featured_image(image_path)
                if featured_id:
                    update_data = {"featured_media": featured_id}
                    update_resp = requests.post(f"{wp_json_url}/posts/{post_id}", json=update_data, headers=headers, timeout=10)
                    if update_resp.status_code == 200:
                        logging.info(f"Featured image set: {featured_id}")
                        print(f"Featured image set: {featured_id}")
                # Log to stats
                with open(STATS_FILE, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([post_id, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), post_link, article_topic, ', '.join(categories)])
            else:
                logging.error(f"Error creating post: {response.status_code} - {response.text}")
                print(f"Error creating post: {response.text}")
        except Exception as e:
            logging.error(f"Error posting to WordPress: {e}")
            print(f"Error posting to WordPress: {e}")

print("Script complete! Check your WordPress dashboard for the magic! [Completed at 09:39 PM +01, Oct 25, 2025]")
logging.info("Script complete")
