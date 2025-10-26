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

# Setup logging with current date and time
logging.basicConfig(filename='autopost.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
print("🚀 Kicking off the blog post generator with some eco-energy vibes! 🌞 [Started at 09:39 PM +01, Oct 25, 2025]")

# Command-line argument parser
parser = argparse.ArgumentParser(description="Generate and optionally post blog content to WordPress with 🔥!")
parser.add_argument("title", nargs='?', help="Article title (e.g., 'Solar Energy'). Optional if --titles-file is used.")
parser.add_argument("--titles-file", help="Text file with multiple titles separated by dots (e.g., 'Title1. Title2. Title3.')")
parser.add_argument("--post-to-wp", action="store_true", help="Auto-post to WordPress like a boss! 🖥️")
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
        print(f"📝 Processing {len(titles)} titles from file: {args.titles_file}")
        logging.info(f"Processing {len(titles)} titles from file: {args.titles_file}")
    except Exception as e:
        logging.error(f"Error reading titles file: {e}")
        print(f"😵 Error reading titles file: {e}")
        sys.exit(1)
else:
    titles = [args.title]

# Initialize Groq client
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# Define parameters
word_count_range = (2000, 2500)
output_dir = "images"

# Dynamic category selection
category_map = {
    "solar": ["Renewable Energy", "Smart Devices"],
    "diy": ["DIY Projects", "Sustainable Lifestyle"],
    "eco home": ["Eco Architecture", "Reviews & Guides"],
    "sustainable": ["Sustainable Lifestyle", "Renewable Energy"],
    "smart": ["Smart Devices", "Green Tech & Energy"]
}

# Ensure images directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    logging.info(f"Created directory: {output_dir}")

for article_topic in titles:
    # Validate title length
    if len(article_topic) > 60:
        logging.warning(f"Title '{article_topic}' exceeds 60 characters, truncating to fit.")
        print(f"⚠️ Title '{article_topic}' too long, truncating to 60 chars.")
        article_topic = article_topic[:60].rsplit(' ', 1)[0]

    print(f"📝 Generating a stellar article on '{article_topic}'...")
    keywords_in_title = article_topic.lower().split()
    categories = next(
        (category_map[key] for key in category_map if key in keywords_in_title),
        ["Sustainable Lifestyle", "Renewable Energy"]
    )
    print(f"🏷️ Categories set: {', '.join(categories)}")

    # Generate article content
    content_prompt = f"""
You are an expert eco-tech journalist and senior SEO editor writing for EcoGridLiving
(audience: USA, Canada, EU). Write helpful, trustworthy, and conversion-oriented long-form between 2000-2500 words
content in a friendly, confident tone. The output must be ready to paste directly into the
WordPress HTML editor — no CSS, no Markdown, use emojies, and make the text alive.

REQUIREMENTS:
- LENGTH: 2000-2500 words (minimum 1,900 if needed). 
- OUTPUT: Clean HTML only (<h1>, <h2>, <h3>, <p>, <ul>, <ol>, <li>, <b>, <blockquote>, <table>, <img>, <q>).
- SEO: Primary keyword = '{article_topic}'. Include 3–6 related LSI keywords from your dataset.
  Keyword density: ~1%. Use it in the title, intro, one <h2>, and one FAQ answer.
- META: Add meta title (≤60 chars), meta description (≤155 chars), and slug suggestion.
- DATA SAFETY: No fabricated stats. If data is uncertain, say “(source: recent industry report 2024–2025)”.

STRUCTURE:
1. HTML comment block at top: meta title, meta description, slug, and word count.
2. <h2> — Article title (use '{article_topic}'+ benefit or hook).
4. Intro (150–220 words): engaging hook with '{article_topic}' early.
5. <h2> Why This Matters — 200–300 words.
6. <h2> Key Benefits — bulleted list (120–180 words, include emojis).
7. <h2> Main Section — depends on format :
   - LISTICLE: 8–12 numbered items, each with <h3>, 2–3 <p>, and one practical tip or warning.
   - HOWTO: 8–12 steps, each with <h3>, short <p>, and checklist (<ol>/<ul>).
   - REVIEW: 5 products/services, each with <h3>, summary, pros, cons, specs, and score (out of 10).
8. <h2> Expert Tips — 150–200 words.
9. <h2> Case Study or Example — 150–200 words (mention USA, Canada, or EU context).
10. <h2> Conclusion — 120–180 words.
11. <h2> FAQ — 3 questions with '{article_topic}' in at least one. Include valid JSON-LD FAQ schema (use qoutes tag in html).

STYLE:
- Use <p> for paragraphs (2–3 sentences max).
- Use <b> for emphasis.
- Include 4–8 bullet lists total.
- Add 1–2 emojis per major section (not in meta/title).
- Include at least one <blockquote> and one numbered checklist.
- Output must be valid, clean HTML only — no inline styles, CSS, or Markdown.

Now write the article for:
TOPIC: {article_topic}
MAIN_KEYWORD: {article_topic}
FORMAT: {article_topic}
TONE: friendly expert
"""

    
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": content_prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=6000,
        )
        article_content = chat_completion.choices[0].message.content
        logging.info(f"Article generated successfully for '{article_topic}'")
        print("🎉 Article generated! Ready to make it shine with some HTML magic...")
    except Exception as e:
        logging.error(f"Error generating article for '{article_topic}': {e}")
        print(f"😵 Oops, something broke: Error generating article: {e}")
        continue

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
        seo_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": seo_prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=300,
        )
        seo_content = seo_completion.choices[0].message.content
        meta_title_match = re.search(r'Meta Title:\s*(.+)', seo_content)
        meta_description_match = re.search(r'Meta Description:\s*(.+?)\s*Keywords:', seo_content, re.DOTALL)
        keywords_match = re.search(r'Keywords:\s*(.+)', seo_content)
        
        meta_title = meta_title_match.group(1).strip()[:60] if meta_title_match else f"{article_topic}: Eco Benefits & Tips"[:60]
        meta_description = meta_description_match.group(1).strip()[:160] if meta_description_match else f"Explore {article_topic.lower()} for sustainable homes in USA, Canada, EU. Benefits, tips, eco guides."[:160]
        keywords = keywords_match.group(1).strip() if keywords_match else f"{article_topic.lower()}, sustainable living, eco-friendly technology"
        
        logging.info(f"SEO metadata generated for '{article_topic}'")
        print("📈 SEO metadata generated!")
        logging.info(f"Meta title length: {len(meta_title)} chars, Meta description length: {len(meta_description)} chars")
    except Exception as e:
        logging.error(f"Error generating SEO metadata for '{article_topic}': {e}")
        print(f"😵 Error generating SEO metadata: {e}")
        meta_title = f"{article_topic}: Eco Benefits & Tips"[:60]
        meta_description = f"Explore {article_topic.lower()} for sustainable homes in USA, Canada, EU. Benefits, tips, eco guides."[:160]
        keywords = f"{article_topic.lower()}, sustainable living, eco-friendly technology"
        logging.info(f"Fallback SEO metadata used for '{article_topic}'")

    # Generate slug (permalink) ≤75 characters
    slug = re.sub(r'[^a-z0-9-]', '', article_topic.lower().replace(' ', '-'))[:75]
    logging.info(f"Generated slug: {slug} ({len(slug)} chars)")

    # Set post title to the original topic (for <h1> and post title)
    post_title = article_topic

    # Add HTML formatting (H1, H2, H3, tables, line breaks)
    def format_article_content(content, title):
        # Add H1 for title (once)
        formatted_content = ""
        
        # Split into lines and add line breaks after periods
        lines = content.split('\n')
        formatted_lines = []
        for line in lines:
            # Skip any unwanted image prompt or caption-like text (expanded filter)
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
            # Add H2 for question-based subheadings (e.g., "What Is...", "How Does...")
            elif re.match(r'^(What|How|Why|When|Where|Who) .+\?$', line, re.IGNORECASE):
                formatted_lines.append(f"<h2>{line}</h2>")
            # Add H3 for other subheadings or key points
            elif re.match(r'^[A-Z][a-zA-Z\s]+$', line) and len(line.split()) > 1:
                formatted_lines.append(f"<h3>{line}</h3>")
            else:
                # Add line breaks after periods
                line = re.sub(r'\.(\s+)', r'.<br>\n', line)
                formatted_lines.append(line)
        
        # Convert pipe-separated tables to HTML
        table_pattern = r'^\|(.+?)\|\s*$'
        table_lines = []
        in_table = False
        for line in formatted_lines:
            if re.match(table_pattern, line):
                if not in_table:
                    table_lines.append("<table border='1' style='border-collapse: collapse; width: 100%;'>")
                    in_table = True
                # Split table row
                cells = [cell.strip() for cell in line.strip('|').split('|')]
                if cells[0].lower().replace(' ', '') in ['benefit', 'feature', 'step', 'cost', 'item']:  # Header row
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

    # Generate image (try Pollinations.ai, fallback to DeepAI)
    image_filename = f"{re.sub(r'[^a-zA-Z0-9 ]', '', article_topic).lower().replace(' ', '_')}.jpg"
    image_path = os.path.join(output_dir, image_filename)
    image_prompt = f"""
    Beautiful modern eco-friendly scene that visually represents {article_topic}. 
    Clean composition, soft natural daylight, realistic photography style, minimal design, 
    focus on sustainable technology and nature harmony — no people, no text, no hands, no close-up faces, no keyboards. 
    Warm natural colors, slightly cinematic lighting, eco lifestyle magazine aesthetic, 16:9 aspect ratio.
    """

    def generate_image(prompt, width=1920, height=1080):
        # Try Pollinations.ai first
        try:
            print("🖼️ Generating image with Pollinations.ai...")
            params = {"width": width, "height": height, "nologo": True, "seed": random.randint(1, 999999999), "model": "flux"}
            encoded_prompt = urllib.parse.quote(prompt)
            url = f"https://image.pollinations.ai/prompt/{encoded_prompt}"
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                with open(image_path, "wb") as f:
                    f.write(response.content)
                # Remove all metadata
                img = Image.open(image_path)
                img_data = list(img.getdata())
                new_img = Image.new(img.mode, img.size)
                new_img.putdata(img_data)
                new_img.save(image_path, "JPEG", quality=95)
                logging.info(f"All metadata removed from {image_path}")
                print(f"🧹 All metadata removed from {image_path}")
                return True
            else:
                logging.error(f"Pollinations.ai failed: {response.status_code} - {response.text}")
                print(f"😓 Pollinations.ai failed: {response.status_code} - {response.text}")
        except Exception as e:
            print("🖼️ Generating image with Pollinations.ai...")
            params = {"width": width, "height": height, "nologo": True, "seed": random.randint(1, 999999999), "model": "flux"}
            encoded_prompt = urllib.parse.quote(prompt)
            url = f"https://image.pollinations.ai/prompt/{encoded_prompt}"
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                with open(image_path, "wb") as f:
                    f.write(response.content)
                # Remove all metadata
                img = Image.open(image_path)
                img_data = list(img.getdata())
                new_img = Image.new(img.mode, img.size)
                new_img.putdata(img_data)
                new_img.save(image_path, "JPEG", quality=95)
                logging.info(f"All metadata removed from {image_path}")
                print(f"🧹 All metadata removed from {image_path}")
            logging.error(f"Pollinations.ai error: {e}")
            print(f"😓 Pollinations.ai error: {e}")

        # Fallback to DeepAI
        try:
            print("🔄 Trying DeepAI as fallback...")
            deepai_key = os.environ.get("DEEPAI_API_KEY", "your_deepai_api_key")
            if deepai_key == "your_deepai_api_key":
                logging.error("DeepAI API key missing. Set DEEPAI_API_KEY in environment variables.")
                print("😵 DeepAI API key missing. Set DEEPAI_API_KEY in environment variables.")
                return False
            response = requests.post(
                "https://api.deepai.org/api/text2img",
                data={"text": prompt, "grid_size": "1"},
                headers={"api-key": deepai_key},
                timeout=15
            )
            if response.status_code == 200 and "output_url" in response.json():
                img_url = response.json()["output_url"]
                img_response = requests.get(img_url, timeout=15)
                if img_response.status_code == 200:
                    with open(image_path, "wb") as f:
                        f.write(img_response.content)
                    # Remove all metadata
                    img = Image.open(image_path)
                    img_data = list(img.getdata())
                    new_img = Image.new(img.mode, img.size) 
                    new_img.putdata(img_data)
                    new_img.save(image_path, "JPEG", quality=95)
                    logging.info(f"All metadata removed from {image_path}")
                    print(f"🧹 All metadata removed from {image_path}")
                    return True
                else:
                    logging.error(f"DeepAI image download failed: {img_response.status_code}")
                    print(f"😓 DeepAI image download failed: {img_response.status_code}")
            else:
                logging.error(f"DeepAI API failed: {response.status_code} - {response.text}")
                print(f"😓 DeepAI API failed: {response.status_code} - {response.text}")
            return False
        except Exception as e:
            logging.error(f"DeepAI error: {e}")
            print(f"😓 DeepAI error: {e}")
            return False

    image_result = generate_image(image_prompt)
    if not image_result:
        print(f"⚠️ Image generation failed for '{article_topic}'. Check the prompt and generate manually at https://pollinations.ai or https://deepai.org.")
        logging.warning(f"Image generation failed for '{article_topic}'")

    # Save to .txt (plain text for reference, including prompt)
    output_filename = f"{slug}_blog_post.txt"
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
        print(f"💾 Blog post saved to {output_filename}")
    except Exception as e:
        logging.error(f"Error saving file for '{article_topic}': {e}")
        print(f"😵 Error saving file: {e}")
        continue

    # Auto-post to WordPress
    if args.post_to_wp:
        print(f"🌐 Starting WordPress auto-post for '{article_topic}'... Let’s get this live! 🚀")
        logging.info(f"Starting WordPress auto-post for '{article_topic}'")
        
        # WordPress config
        wp_site_url = os.environ.get("WP_SITE_URL")
        wp_username = os.environ.get("WP_USERNAME")
        wp_app_password = os.environ.get("WP_APP_PASSWORD")
        
        if not all([wp_site_url, wp_username, wp_app_password]):
            error_msg = "Error: Set WP_SITE_URL, WP_USERNAME, and WP_APP_PASSWORD in environment variables.\nExample (PowerShell):\n$env:WP_SITE_URL = 'https://ecogridliving.lovestoblog.com'\n$env:WP_USERNAME = 'abdallahh'\n$env:WP_APP_PASSWORD = 'cR8P OrCy AkEI PWIZ YHQH A2rx'"
            logging.error(error_msg)
            print(f"😵 {error_msg}")
            continue
        
        wp_json_url = f"{wp_site_url.rstrip('/')}/wp-json/wp/v2"
        auth_str = f"{wp_username}:{wp_app_password}"
        b64_auth = base64.b64encode(auth_str.encode()).decode()
        headers = {
            "Authorization": f"Basic {b64_auth}",
            "Content-Type": "application/json"
        }
        
        # Test API connectivity
        try:
            test_response = requests.get(f"{wp_json_url}/posts", headers=headers, timeout=10)
            if test_response.status_code != 200:
                error_msg = f"Error connecting to WordPress API: {test_response.status_code} - {test_response.text}\nCheck: 1) WP_SITE_URL is correct, 2) Username/Application Password valid, 3) User has Editor/Administrator role, 4) WordPress.com plan (Business or higher for write access)."
                logging.error(error_msg)
                print(f"😵 {error_msg}")
                continue
            logging.info("WordPress API connectivity test successful")
            print("✅ WordPress API is ready to rock!")
        except Exception as e:
            error_msg = f"Error testing WordPress API: {e}\nCheck: 1) Internet connection, 2) WP_SITE_URL, 3) Server config, 4) WordPress.com plan."
            logging.error(error_msg)
            print(f"😵 {error_msg}")
            continue
        
        def get_or_create_category(name):
            try:
                response = requests.get(f"{wp_json_url}/categories?search={urllib.parse.quote(name)}", headers=headers, timeout=10)
                if response.status_code == 200 and response.json():
                    return response.json()[0]["id"]
                create_data = {"name": name, "slug": re.sub(r'[^a-z0-9-]', '', name.lower().replace(' ', '-'))}
                create_resp = requests.post(f"{wp_json_url}/categories", json=create_data, headers=headers, timeout=10)
                if create_resp.status_code == 201:
                    logging.info(f"Created category: {name}")
                    print(f"🏷️ Created category: {name}")
                    return create_resp.json()["id"]
                error_msg = f"Error creating category '{name}': {create_resp.status_code} - {create_resp.text}"
                logging.error(error_msg)
                print(f"😵 {error_msg}")
                return None
            except Exception as e:
                error_msg = f"Error with category '{name}': {e}"
                logging.error(error_msg)
                print(f"😵 {error_msg}")
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
                            print(f"🏷️ Created tag: {tag}")
                except Exception as e:
                    error_msg = f"Error with tag '{tag}': {e}"
                    logging.error(error_msg)
                    print(f"😵 {error_msg}")
            return tag_ids
        
        def upload_featured_image(image_path):
            if not os.path.exists(image_path):
                error_msg = f"Warning: Image not found at {image_path}. Skipping featured image."
                logging.warning(error_msg)
                print(f"⚠️ {error_msg}")
                return None
            try:
                with open(image_path, "rb") as img_file:
                    data = {
                        "caption": "",  # Explicitly set to empty to avoid captions
                        "description": article_topic,
                        "alt_text": ""  # Cleaner alt text, no extra words
                    }
                    files = {"file": (image_filename, img_file, "image/jpeg")}
                    media_headers = {
                        "Authorization": headers["Authorization"],
                        "Content-Disposition": f'attachment; filename={image_filename}'
                    }
                    response = requests.post(f"{wp_json_url}/media", data=data, files=files, headers=media_headers, timeout=15)
                    if response.status_code == 201:
                        media_id = response.json()["id"]
                        logging.info(f"Featured image uploaded: {media_id}")
                        print(f"🖼️ Featured image uploaded: {media_id}")
                        return media_id
                    error_msg = f"Error uploading image: {response.status_code} - {response.text}"
                    logging.error(error_msg)
                    print(f"😵 {error_msg}")
                    return None
            except Exception as e:
                error_msg = f"Error uploading image: {e}"
                logging.error(error_msg)
                print(f"😵 {error_msg}")
                return None
        
        # Create post
        post_data = {
            "title": article_topic,
            "content": article_content,
            "excerpt": meta_description,
            "slug": slug,
            "status": "publish",  # Change to "publish" if desired
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
                success_msg = f"Post created successfully! ID: {post_id} | View: {wp_site_url}/wp-admin/post.php?post={post_id}&action=edit"
                logging.info(success_msg)
                print(f"🎉 {success_msg}")
                featured_id = upload_featured_image(image_path)
                if featured_id:
                    update_data = {"featured_media": featured_id}
                    update_resp = requests.post(f"{wp_json_url}/posts/{post_id}", json=update_data, headers=headers, timeout=10)
                    if update_resp.status_code == 200:
                        logging.info(f"Featured image set: {featured_id}")
                        print(f"🖼️ Featured image set: {featured_id}")
                    else:
                        error_msg = f"Error setting featured image: {update_resp.status_code} - {update_resp.text}"
                        logging.error(error_msg)
                        print(f"😵 {error_msg}")
            else:
                error_msg = f"Error creating post: {response.status_code} - {response.text}\nPossible issues: 1) Invalid credentials, 2) Insufficient permissions, 3) WordPress.com free plan limits write access (upgrade to Business plan)."
                logging.error(error_msg)
                print(f"😵 {error_msg}")
        except Exception as e:
            error_msg = f"Error posting to WordPress: {e}\nCheck: 1) Internet connection, 2) WP_SITE_URL, 3) Server config, 4) WordPress.com plan (Business or higher for write access)."
            logging.error(error_msg)
            print(f"😵 {error_msg}")

print("🏁 Script complete! Check your WordPress dashboard for the magic! ✨ [Completed at 09:39 PM +01, Oct 25, 2025]")
logging.info("Script complete")
"""hello world!"""