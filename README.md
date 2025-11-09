# AI Blog Post Generator & Auto-Publisher for WordPress  
**`autopost.py` – Fully Automated SEO-Optimized AI/ML Tutorial Generator with Image & WordPress Publishing**

![Python](https://img.shields.io/badge/python-3.9%2B-blue?logo=python)  
![WordPress](https://img.shields.io/badge/WordPress-Compatible-brightgreen?logo=wordpress)  
![AI Powered](https://img.shields.io/badge/AI%20Powered-LLM%20%2B%20Image%20Gen-orange)  
![WebP](https://img.shields.io/badge/Images-WebP%20Optimized-lightgrey)  
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## Why This Script is Cool

This **end-to-end content automation engine** transforms a **single line of text (a title)** into a **fully published, SEO-optimized, 2,000+ word AI/ML tutorial** — complete with:

- **Rich HTML article** (headings, code blocks, tables, schema)
- **AI-generated 1920×1080 WebP image** (via Pollinations.ai + PIL)
- **Smart category hierarchy** (parent → child auto-creation)
- **Rank Math SEO metadata** (title, description, focus keywords)
- **WordPress auto-posting** with backdated publishing
- **Triple LLM failover** (Groq → Groq → OpenRouter)
- **Batch processing** from `.txt` file
- **Full audit trail** (`log` + `CSV` stats)

> **Use Case:** Scale high-quality, educational AI content at **zero manual effort**.

---

## Features

| Feature | Description |
|--------|-----------|
| **Multi-LLM Fallback** | Groq (2 keys) → OpenRouter (`mistral-7b-instruct:free`) |
| **SEO-Rich HTML** | `<h1>`, `<pre><code>`, tables, FAQ schema, emoji |
| **WebP Image Pipeline** | FLUX model → 1920×1080 → WebP (80% quality) |
| **WordPress REST API** | Auto-creates categories, tags, uploads media |
| **Rank Math Integration** | Sets `rank_math_title`, `description`, `focus_keyword` |
| **Backdated Publishing** | Random date in last 30 days for organic feel |
| **Batch Mode** | Process 100+ titles from a `.txt` file |
| **Logging & Stats** | `autopost.log` + `published_articles.csv` |

---

## Setup Instructions

### 1. **Download or Clone**
```bash
git clone https://github.com/dev0lcyber/wp-auto-blog-post.git
cd wp-auto-blog-post
```

2. Install Dependencies
bashpip install requests pillow groq

All standard libraries included (csv, logging, argparse, etc.)


3. Get Required API Keys

























ServiceLinkNotesGroqconsole.groq.comCreate 2 API keys for redundancyOpenRouteropenrouter.aiUse free mistralai/mistral-7b-instruct:freeWordPress App PasswordWP Admin → Users → Your Profile → App PasswordsMust be Administrator

4. Edit autopost.py – Insert Your Credentials
pythonGROQ_API_KEY = "gsk_your_first_key_here"
GROQ_API_KEY2 = "gsk_your_second_key_here"
OPENROUTER_API_KEY = "sk-or-v1-your-openrouter-key"
WP_SITE_URL = "https://yourblog.com/"
WP_USERNAME = "your_wp_username"
WP_APP_PASSWORD = "abcd efgh ijkl mnop qrst uvwx"  # 24 characters, space-separated

Security Tip: Never commit real keys. Use local-only file or .env.


Project Structure
textai-blog-autoposter/
│
├── autopost.py                  ← Main script (this file)
├── blogs/                       ← All generated content
│   └── how-to-build-neural-network/
│       ├── how-to-build-neural-network.webp
│       └── how-to-build-neural-network_blog_post.txt
├── published_articles.csv       ← Published post stats
├── autopost.log                 ← Full runtime log
├── titles.txt                   ← (Optional) Batch input
└── README.md                    ← You're reading it!

Usage
Run a Single Article
bashpython autopost.py "How to Fine-Tune LLMs with LoRA"
Batch Process from File
bashpython autopost.py --titles-file titles.txt
Auto-Publish to WordPress
bashpython autopost.py "Your Title" --post-to-wp
Full Batch + Publish
bashpython autopost.py --titles-file titles.txt --post-to-wp

titles.txt Format (Batch Input)
txtHow to Build a Neural Network in Python.
Fine-Tuning LLMs with LoRA: A Step-by-Step Guide.
Create a Flutter AI Chatbot with Gemini API.
Understanding Gradient Descent Visually.

Titles separated by . (dot) — can be on new lines.


Output Per Article
Each article gets a dedicated folder in blogs/:
Example: blogs/neural-network-python/
textneural-network-python_blog_post.txt
neural-network-python.webp

Inside _blog_post.txt
txtArticle Content:
<!-- META: Title, Desc, Slug, Word Count -->
<h1>Master Neural Networks in Python for Real AI Projects</h1>
<p>Unlock the power of deep learning...</p>
...

Categories:
ML Tutorials, Python for ML

Main Image:
Filename: neural-network-python.webp
Path: blogs/neural-network-python/neural-network-python.webp
Prompt: modern sci-fi tech scene that visually represents neural networks.

SEO Metadata:
Meta Title: Neural Networks in Python: Complete 2025 Guide
Meta Description: Learn to build, train, and deploy neural networks in Python with code examples, tips, and real projects. Perfect for beginners and pros.
Slug: neural-network-python
Keywords: neural network python, deep learning tutorial, ai coding

WordPress Publishing (with --post-to-wp)

































ActionDetailsCategoriesAuto-created with parent → child hierarchyTagsFrom SEO keywords (max 10)Featured ImageWebP uploaded, set as thumbnailSEO Fieldsrank_math_title, description, focus_keywordPublish DateRandom time in last 30 daysStatuspublish immediately

Category Hierarchy (Auto-Managed)
python"AI History" → "AI Fundamentals"
"Beginner Projects" → "ML Tutorials"
"Future Outlook" → "AI Trends & Insights"
"Case Studies" → "Ethical AI"

Script creates missing categories with correct parent.


Image Generation Pipeline

Model: FLUX (via Pollinations.ai)
Resolution: 1920×1080
Seed: Random per image
Output:.webp (80% quality, ~300–600 KB)
No logo (nologo=True)


Logging & Stats
autopost.log
text2025-10-25 21:39:12,123 - INFO - Kicking off...
2025-10-25 21:39:45,789 - INFO - Article generated: How to Build...
2025-10-25 21:40:10,234 - INFO - WebP image saved: blogs/...
2025-10-25 21:41:05,567 - INFO - Post ID 1234: https://driouich.me/...
published_articles.csv
csvID,Date,Link,Title,Categories
1234,2025-10-25 21:41,https://driouich.me/neural-network-python,How to Build a Neural Network in Python,"ML Tutorials, Python for ML"

Article Structure (Generated)
html<!-- META: ... -->
<h1>Master {Topic} for Practical AI Skills</h1>
<p>Engaging intro with keyword...</p>

<h2>Prerequisites</h2>
<ul><li>Python 3.9+</li>...</ul>

<h2>Why This Matters</h2>
<p>Real-world impact...</p>

<h2>Key Benefits</h2>
<ul><li>Learn by doing</li>...</ul>

<h2>Step-by-Step Guide</h2>
<h3>Step 1: Install Dependencies</h3>
<pre><code class="language-python">pip install torch torchvision</code></pre>

...

<h2>FAQ</h2>
<h3>What is {topic}?</h3>
<p>Answer...</p>

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "FAQPage",
  "mainEntity": [...]
}
</script>

Best Practices

Use clear, benefit-driven titles (<60 chars)
Run in screen/tmux for long batches
Monitor autopost.log for API issues
Install Rank Math SEO plugin on WordPress
Schedule via cron:

bash# Daily at 3 AM
0 3 * * * cd /path/to/script && python autopost.py --titles-file daily_titles.txt --post-to-wp >> /var/log/autopost_cron.log 2>&1

Troubleshooting

























IssueFix401 UnauthorizedCheck WP App Password (24 chars, spaces)Image failedPollinations.ai may be down → retryGroq rate limitScript auto-falls back to OpenRouterCategories not createdEnsure WP user is Administrator

Roadmap / Future Enhancements

 Add DeepAI / Stable Diffusion image fallback
 Auto-generate internal links to past articles
 YouTube script + thumbnail generator
 Auto-post to Twitter/X threads
 Ping Google Search Console
Streamlit GUI for non-coders
Docker container


Safety & Ethics

No fake stats – uses “industry report 2024–2025” if unsure
Educational focus only
No spam – high-value, human-like tutorials
Transparent sourcing


License
textMIT License

Copyright (c) 2025 Abdallah Driouich

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software...

Made with Love for AI Education

"Automate the boring. Teach the world AI."
