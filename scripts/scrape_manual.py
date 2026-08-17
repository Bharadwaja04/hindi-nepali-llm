#!/usr/bin/env python3
"""Comprehensive manual data scraper for Hindi and Nepali.

Sources (all original-language, NOT translations):
  - RSS feeds from Hindi/Nepali news sites (structured, clean text)
  Hindi news: BBC Hindi, NDTV Hindi, Aaj Tak, Dainik Jagran, Amar Ujala
  Nepali news: Setopati, Onlinekhabar, Ratopati, Kantipur
  - Hindi literature: kavitakosh.org (large Hindi poetry collection)
  - Nepali literature: sahityaras (already partially collected)
  - Wikisource (Hindi API + Nepali HTML)
  - Wikipedia article extraction (manual curation of specific topics)

All tagged source: manual for the 20% requirement.
"""

import argparse
import json
import os
import re
import time
import urllib.parse
import urllib.request
from bs4 import BeautifulSoup
from xml.etree import ElementTree


# =====================================================================
# RSS FEED SCRAPER (news sites — clean, structured text)
# =====================================================================
HINDI_RSS_FEEDS = [
    # (name, url)
    ("bbc_hindi", "https://feeds.bbci.co.uk/hindi/rss.xml"),
    ("ndtv_hindi", "https://khabar.ndtv.com/rss/news"),
    ("aajtak_top", "https://www.aajtak.in/rssfeeds/?id=home"),
    ("aajtak_india", "https://www.aajtak.in/rssfeeds/?id=india"),
    ("aajtak_world", "https://www.aajtak.in/rssfeeds/?id=world"),
    ("aajtak_sports", "https://www.aajtak.in/rssfeeds/?id=sports"),
    ("aajtak_entertainment", "https://www.aajtak.in/rssfeeds/?id=entertainment"),
    ("aajtak_business", "https://www.aajtak.in/rssfeeds/?id=business"),
    ("dainik_jagran_top", "https://www.jagran.com/rss/top-news.xml"),
    ("dainik_jagran_nation", "https://www.jagran.com/rss/nation.xml"),
    ("dainik_jagran_world", "https://www.jagran.com/rss/world.xml"),
    ("dainik_jagran_business", "https://www.jagran.com/rss/business.xml"),
    ("dainik_jagran_sports", "https://www.jagran.com/rss/sports.xml"),
    ("amar_ujala_top", "https://www.amarujala.com/rss/top-news"),
    ("amar_ujala_india", "https://www.amarujala.com/rss/india"),
    ("amar_ujala_world", "https://www.amarujala.com/rss/world"),
    ("amar_ujala_business", "https://www.amarujala.com/rss/business"),
    ("amar_ujala_sports", "https://www.amarujala.com/rss/sports"),
    ("amar_ujala_entertainment", "https://www.amarujala.com/rss/entertainment"),
    ("hindustan_top", "https://www.livehindustan.com/rss/top-news.xml"),
    ("hindustan_nation", "https://www.livehindustan.com/rss/nation.xml"),
    ("patrika_india", "https://www.patrika.com/rss/india-news.xml"),
    ("patrika_world", "https://www.patrika.com/rss/world-news.xml"),
    ("patrika_business", "https://www.patrika.com/rss/business.xml"),
    ("patrika_sports", "https://www.patrika.com/rss/sports.xml"),
    ("navbharat_top", "https://navbharattimes.indiatimes.com/rssfeedstopstories.cms"),
    (
        "navbharat_entertainment",
        "https://navbharattimes.indiatimes.com/rssfeeds/entertainment.cms",
    ),
    ("news18_hindi", "https://hindi.news18.com/rss/khabar/"),
    ("zee_hindi", "https://zeenews.india.com/hindi/rss.xml"),
    ("republic_hindi", "https://hindi.republicworld.com/rss/feed"),
]

NEPALI_RSS_FEEDS = [
    ("setopati", "https://www.setopati.com/rss"),
    ("onlinekhabar", "https://www.onlinekhabar.com/feed"),
    ("ratopati", "https://ratopati.com/rss"),
    ("kantipur_top", "https://kathmandupost.com/rss"),
    ("kantipur_news", "https://ekantipur.com/rss"),
    ("gorkhapatra", "https://gorkhapatraonline.com/rss"),
    ("himalayan_times", "https://thehimalayantimes.com/feed"),
    ("republica_nepal", "https://myrepublica.nagariknetwork.com/feed"),
    ("annapurna_post", "https://annapurnapost.com/feed"),
    ("nagarik_news", "https://nagarik.nagariknetwork.com/feed"),
    ("mero_desh", "https://merodesh.com/feed"),
    ("ariko_post", "https://www.arikopost.com/feed"),
    ("newa_dine", "https://newadine.com/feed"),
    ("bahrabari", "https://bahrabari.com/feed"),
    ("ujyaalo", "https://ujyaaloonline.com/feed"),
]


def fetch_url(url, timeout=30):
    """Fetch URL content with proper headers."""
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; LMA-Project/1.0; educational)"
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def parse_rss_feed(feed_url, feed_name, lang):
    """Parse an RSS/Atom feed and return list of (title, link, description) tuples."""
    try:
        raw = fetch_url(feed_url, timeout=20)
        root = ElementTree.fromstring(raw)

        items = []
        # RSS 2.0
        for item in root.iter("item"):
            title = item.findtext("title", default="")
            link = item.findtext("link", default="")
            desc = item.findtext("description", default="")
            items.append((title, link, desc))

        # Atom
        if not items:
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            for entry in root.findall(".//atom:entry", ns):
                title = entry.findtext("atom:title", default="", namespaces=ns)
                link_elem = entry.find("atom:link", namespaces=ns)
                link = link_elem.get("href", "") if link_elem is not None else ""
                summary = entry.findtext("atom:summary", default="", namespaces=ns)
                items.append((title, link, summary))

        return items
    except Exception as e:
        print(f"  [rss] {feed_name}: error - {str(e)[:80]}")
        return []


def clean_html_text(html_text):
    """Strip HTML to plain text."""
    if not html_text:
        return ""
    soup = BeautifulSoup(html_text, "lxml")
    for tag in soup(["script", "style", "nav", "table", "form"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def scrape_article(url, timeout=15):
    """Fetch and extract main text from a news article URL."""
    try:
        html = fetch_url(url, timeout=timeout).decode("utf-8", errors="replace")
        soup = BeautifulSoup(html, "lxml")

        # Remove non-content
        for tag in soup(
            [
                "script",
                "style",
                "nav",
                "header",
                "footer",
                "aside",
                "form",
                "input",
                "select",
                "iframe",
                "noscript",
            ]
        ):
            tag.decompose()

        # Try common article content selectors
        content = None
        for selector in [
            (
                "div",
                {"class": re.compile(r"article|content|story|post|entry|main", re.I)},
            ),
            ("article", {}),
            ("div", {"id": re.compile(r"article|content|story|post|entry|main", re.I)}),
            ("div", {"class": "article-body"}),
            ("div", {"class": "story-content"}),
            ("div", {"class": "article-content"}),
            ("div", {"itemprop": "articleBody"}),
        ]:
            tag_name, attrs = selector
            content = soup.find(tag_name, attrs)
            if content and len(content.get_text()) > 200:
                break
            content = None

        if content is None:
            # fallback: get all <p> tags
            paragraphs = soup.find_all("p")
            if paragraphs:
                content = BeautifulSoup("", "lxml").new_tag("div")
                for p in paragraphs:
                    content.append(p)

        if content is None:
            return None

        text = content.get_text(separator="\n")
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()
    except:
        return None


def scrape_rss_feeds(feed_list, lang, outdir, max_articles_per_feed=100):
    """Scrape articles from RSS feeds."""
    os.makedirs(outdir, exist_ok=True)
    all_records = []
    total_chars = 0
    n_out = 0

    shard_idx = 0
    shard_path = os.path.join(outdir, f"{lang}_news_{shard_idx:04d}.jsonl")
    f_out = open(shard_path, "w", encoding="utf-8")

    for feed_name, feed_url in feed_list:
        print(f"  [{feed_name}] fetching feed...", flush=True)
        items = parse_rss_feed(feed_url, feed_name, lang)
        print(f"  [{feed_name}] {len(items)} articles in feed", flush=True)

        feed_n = 0
        for title, link, desc in items:
            if feed_n >= max_articles_per_feed:
                break
            if not link:
                continue

            # First try the description (often has full or summary text)
            desc_text = clean_html_text(desc) if desc else ""
            article_text = ""

            if desc_text and len(desc_text) > 200:
                # Description has enough text, use it
                article_text = desc_text
            else:
                # Fetch full article
                article_text = scrape_article(link)
                if not article_text or len(article_text) < 50:
                    if desc_text and len(desc_text) > 50:
                        article_text = desc_text
                    else:
                        continue

            # Check Devanagari ratio
            deva = sum(1 for c in article_text if 0x0900 <= ord(c) <= 0x097F)
            if len(article_text) > 0 and deva / max(len(article_text), 1) < 0.10:
                continue

            rec = {
                "text": article_text,
                "source": "manual",
                "lang": lang,
                "dataset": f"news_{feed_name}",
                "title": title,
                "url": link,
                "n_chars": len(article_text),
                "n_bytes": len(article_text.encode("utf-8")),
            }
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_out += 1
            total_chars += len(article_text)
            feed_n += 1

            if n_out % 500 == 0 and n_out > 0:
                f_out.close()
                shard_idx += 1
                shard_path = os.path.join(outdir, f"{lang}_news_{shard_idx:04d}.jsonl")
                f_out = open(shard_path, "w", encoding="utf-8")

            time.sleep(0.3)  # be nice

        print(
            f"  [{feed_name}] saved {feed_n} articles, running total: {n_out} ({total_chars / 1e6:.1f} MB)",
            flush=True,
        )

    f_out.close()
    print(f"\n  RSS DONE: {n_out} articles, {total_chars / 1e6:.1f} MB", flush=True)
    return n_out, total_chars


# =====================================================================
# WIKIPEDIA ARTICLE SCRAPER (curated topic lists)
# =====================================================================
HINDI_WIKI_TOPICS = [
    "भारत",
    "हिन्दी",
    "नेपाली_भाषा",
    "संस्कृति",
    "इतिहास",
    "भूगोल",
    "विज्ञान",
    "गणित",
    "खगोल_विज्ञान",
    "जीव_विज्ञान",
    "रसायन_विज्ञान",
    "भौतिकी",
    "कंप्यूटर",
    "प्रौद्योगिकी",
    "चिकित्सा",
    "कला",
    "संगीत",
    "नृत्य",
    "चित्रकला",
    "वास्तुकला",
    "साहित्य",
    "कविता",
    "उपन्यास",
    "कहानी",
    "नाटक",
    "लोककथाएँ",
    "धर्म",
    "हिन्दू_धर्म",
    "बौद्ध_धर्म",
    "जैन_धर्म",
    "इस्लाम",
    "ईसाई_धर्म",
    "सिख_धर्म",
    "दर्शन",
    "योग",
    "राजनीति",
    "अर्थशास्त्र",
    "समाजशास्त्र",
    "मनोविज्ञान",
    "शिक्षा",
    "कृषि",
    "उद्योग",
    "वाणिज्य",
    "पर्यावरण",
    "जलवायु",
    "वन",
    "पहाड़",
    "नदी",
    "समुद्र",
    "मरुस्थल",
    "जंगल",
    "जानवर",
    "पक्षी",
    "मछली",
    "कीट",
    "फूल",
    "पेड़",
    "फल",
    "सब्जी",
    "मसाला",
    "खाना",
    "वस्त्र",
    "घर",
    "परिवार",
    "विवाह",
    "त्योहार",
    "दीवाली",
    "होली",
    "दशहरा",
    "रक्षाबंधन",
    "ईद",
    "क्रिसमस",
    "गुरु_नानक_जयंती",
    "बुद्ध_जयंती",
    "महावीर_जयंती",
    "स्वतंत्रता_दिवस",
    "गणतंत्र_दिवस",
    "गांधी_जयंती",
    "बाल_दिवस",
    "महिला_दिवस",
    "शिक्षक_दिवस",
    "विज्ञान_दिवस",
    "पर्यावरण_दिवस",
    "खेल",
    "क्रिकेट",
    "फुटबॉल",
    "हॉकी",
    "टेनिस",
    "बैडमिंटन",
    "कबड्डी",
    "कुश्ती",
    "मुक्केबाजी",
    "तैराकी",
    "धनुर्विद्या",
    "भारत_का_संविधान",
    "भारत_का_इतिहास",
    "मुग़ल_साम्राज्य",
    "ब्रिटिश_राज",
    "स्वतंत्रता_आंदोलन",
    "गांधी",
    "नेहरू",
    "पटेल",
    "बोस",
    "भगत_सिंह",
    "तिलक",
    "गोखले",
    "लाल_बहादुर_शास्त्री",
    "इंदिरा_गांधी",
    "राजीव_गांधी",
    "अटल_बिहारी_वाजपेयी",
    "एपीजे_अब्दुल_कलाम",
    "प्रणब_मुखर्जी",
    "अमर्त्य_सेन",
    "रवींद्रनाथ_ठाकुर",
    "प्रेमचंद",
    "जयशंकर_प्रसाद",
    "सूर्यकांत_त्रिपाठी_निराला",
    "जैशंकर_प्रसाद",
    "मैथिली_शरण_गुप्त",
    "रामधारी_सिंह_दिनकर",
    "नागार्जुन",
    "हरिवंश_राय_बच्चन",
    "धर्मवीर_भारती",
    "फणीश्वरनाथ_रेणु",
    "अज्ञेय",
    "निर्मल_वर्मा",
    "कमलेश्वर",
    "मोहन_राकेश",
    "भीष्म_साहनी",
    "यशपाल",
    "अमृतलाल_नागर",
]

NEPALI_WIKI_TOPICS = [
    "नेपाल",
    "काठमाडौं",
    "हिमालय",
    "नेपाली_भाषा",
    "नेपालको_इतिहास",
    "नेपालको_भूगोल",
    "नेपालको_संस्कृति",
    "नेपालको_धर्म",
    "हिन्दू_धर्म",
    "बौद्ध_धर्म",
    "नेपाली_साहित्य",
    "नेपाली_कविता",
    "नेपाली_संगीत",
    "नेपाली_खाना",
    "नेपाली_पर्व",
    "दशैं",
    "तिहार",
    "होली",
    "तीज",
    "छठ",
    "बुद्ध_जयन्ती",
    "नेपालका_नदीहरू",
    "नेपालका_तालहरू",
    "नेपालका_हिमचुली",
    "नेपालका_राष्ट्रिय_निकुञ्ज",
    "नेपालको_वन",
    "नेपालको_जनजीवन",
    "गोरखा",
    "पृथ्वीनारायण_शाह",
    "नेपाल_एकीकरण",
    "राणा_शासन",
    "नेपालको_प्रजातन्त्र_आन्दोलन",
    "नेपालको_जनआन्दोलन",
    "नेपालको_संविधान",
    "नेपालको_संसद",
    "नेपालको_सरकार",
    "काठमाडौं_उपत्यका",
    "पोखरा",
    "लुम्बिनी",
    "चितवन",
    "लेकनाथ",
    "धरान",
    "बिराटनगर",
    "भैरहवा",
    "नेपालगन्ज",
    "धनगढी",
    "महेन्द्रनगर",
    "जनकपुर",
    "पाटन",
    "भक्तपुर",
    "नेपाली_कवि",
    "लेखनाथ_पौड्याल",
    "लक्ष्मीप्रसाद_देवकोटा",
    "बालकृष्ण_सम",
    "सिद्धिचरण_श्रेष्ठ",
    "भूपि_शेरचन",
    "मोहन_कोइराला",
    "वैकुण्ठ_शर्मा_निर्झर",
    "गुरुप्रसाद_मैनाली",
    "विश्वेश्वर_प्रसाद_कोइराला",
    "रत्नबहादुर_थापा",
    "श्यामप्रसाद_देवकोटा",
    "माधव_प्रसाद_घिमिरे",
]


def scrape_wikipedia_articles(domain, topics, lang, outdir):
    """Scrape full Wikipedia article text via API."""
    os.makedirs(outdir, exist_ok=True)
    n_out = total_chars = 0

    shard_path = os.path.join(outdir, f"{lang}_wikipedia_{0:04d}.jsonl")
    f_out = open(shard_path, "w", encoding="utf-8")

    for i, topic in enumerate(topics):
        try:
            # Use API extracts (plain text, no HTML)
            params = {
                "action": "query",
                "prop": "extracts",
                "titles": topic,
                "explaintext": "1",
                "exsectionformat": "plain",
                "format": "json",
            }
            url = f"https://{domain}/w/api.php?" + urllib.parse.urlencode(params)
            raw = fetch_url(url, timeout=15)
            data = json.loads(raw.decode("utf-8"))
            pages = data.get("query", {}).get("pages", {})
            for pid, pdata in pages.items():
                extract = pdata.get("extract", "")
                if extract and len(extract) > 200:
                    rec = {
                        "text": extract,
                        "source": "manual",
                        "lang": lang,
                        "dataset": "wikipedia_curated",
                        "title": pdata.get("title", topic),
                        "n_chars": len(extract),
                        "n_bytes": len(extract.encode("utf-8")),
                    }
                    f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_out += 1
                    total_chars += len(extract)
                break
        except:
            pass

        if (i + 1) % 20 == 0:
            print(
                f"  [{i + 1}/{len(topics)}] saved {n_out} articles, {total_chars / 1e6:.1f} MB",
                flush=True,
            )
        time.sleep(0.3)

    f_out.close()
    print(f"  Wikipedia DONE: {n_out} articles, {total_chars / 1e6:.1f} MB", flush=True)
    return n_out, total_chars


# =====================================================================
# LITERATURE SITE SCRAPER
# =====================================================================
def scrape_kavitakosh(lang, outdir, max_pages=500):
    """Scrape kavitakosh.org — large Hindi poetry/literature collection."""
    os.makedirs(outdir, exist_ok=True)
    n_out = total_chars = 0

    shard_path = os.path.join(outdir, f"{lang}_kavitakosh_{0:04d}.jsonl")
    f_out = open(shard_path, "w", encoding="utf-8")

    # Scrape from category pages
    categories = [
        "https://www.kavitakosh.org/hp/कविता_कोष:_सूची_अनुक्रमणिका",
        "https://www.kavitakosh.org/hp/कविता_कोष:_कवियों_की_सूची",
    ]

    poem_urls = set()
    for cat_url in categories:
        try:
            html = fetch_url(cat_url).decode("utf-8", errors="replace")
            soup = BeautifulSoup(html, "lxml")
            content = soup.find("div", id="bodyContent") or soup.body
            if content:
                for a in content.find_all("a", href=True):
                    href = a["href"]
                    if href.startswith("/hp/") and ":" not in href[4:]:
                        full_url = "https://www.kavitakosh.org" + href
                        poem_urls.add(full_url)
        except Exception as e:
            print(f"  [kavitakosh] category error: {str(e)[:80]}")

    print(f"  [kavitakosh] found {len(poem_urls)} poem pages", flush=True)

    for i, url in enumerate(list(poem_urls)[:max_pages]):
        try:
            html = fetch_url(url, timeout=15).decode("utf-8", errors="replace")
            soup = BeautifulSoup(html, "lxml")
            for tag in soup(["script", "style", "nav", "table", "form"]):
                tag.decompose()
            content = soup.find("div", id="bodyContent") or soup.body
            if content:
                text = content.get_text(separator="\n")
                text = re.sub(r"\n{3,}", "\n\n", text)
                text = re.sub(r"[ \t]+", " ", text).strip()
                if len(text) > 100:
                    rec = {
                        "text": text,
                        "source": "manual",
                        "lang": lang,
                        "dataset": "kavitakosh",
                        "url": url,
                        "n_chars": len(text),
                        "n_bytes": len(text.encode("utf-8")),
                    }
                    f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_out += 1
                    total_chars += len(text)
        except:
            pass

        if (i + 1) % 50 == 0:
            print(
                f"  [{i + 1}/{min(len(poem_urls), max_pages)}] saved {n_out}, {total_chars / 1e6:.1f} MB",
                flush=True,
            )
        time.sleep(0.3)

    f_out.close()
    print(f"  Kavitakosh DONE: {n_out} pages, {total_chars / 1e6:.1f} MB", flush=True)
    return n_out, total_chars


# =====================================================================
# WIKISOURCE SCRAPER (fixed — uses prop=text)
# =====================================================================
def scrape_wikisource(domain, lang, outdir, max_pages=5000, use_api=True):
    """Scrape Wikisource for original-language public domain texts."""
    os.makedirs(outdir, exist_ok=True)
    n_out = total_chars = 0

    shard_path = os.path.join(outdir, f"{lang}_wikisource_{0:04d}.jsonl")
    f_out = open(shard_path, "w", encoding="utf-8")

    # List pages
    titles = []
    if use_api:
        apfrom = None
        while len(titles) < max_pages:
            params = {
                "action": "query",
                "list": "allpages",
                "apnamespace": "0",
                "aplimit": "500",
                "apfilterredir": "nonredirects",
            }
            if apfrom:
                params["apfrom"] = apfrom
            try:
                url = f"https://{domain}/w/api.php?" + urllib.parse.urlencode(
                    {**params, "format": "json"}
                )
                data = json.loads(fetch_url(url, timeout=20).decode("utf-8"))
                pages = data.get("query", {}).get("allpages", [])
                titles.extend(p["title"] for p in pages)
                cont = data.get("query-continue", data.get("continue", {}))
                apfrom = cont.get("allpages", {}).get("apfrom") or cont.get("apfrom")
                if not apfrom:
                    break
                time.sleep(0.5)
            except:
                break
    else:
        # HTML fallback for broken APIs (Nepali)
        try:
            url = f"https://{domain}/wiki/Special:AllPages?namespace=0"
            html = fetch_url(url).decode("utf-8", errors="replace")
            soup = BeautifulSoup(html, "lxml")
            content = soup.find("div", id="mw-content-text") or soup.body
            if content:
                for a in content.find_all("a", href=True):
                    href = a["href"]
                    if (
                        href.startswith("/wiki/")
                        and ":" not in href[6:]
                        and "Special" not in href
                    ):
                        title = urllib.parse.unquote(href[6:])
                        titles.append(title)
        except:
            pass

    titles = list(dict.fromkeys(titles))[:max_pages]
    print(f"  [wikisource] {len(titles)} pages to scrape", flush=True)

    for i, title in enumerate(titles):
        try:
            if use_api:
                params = {
                    "action": "parse",
                    "page": title,
                    "prop": "text",
                    "format": "json",
                }
                url = f"https://{domain}/w/api.php?" + urllib.parse.urlencode(params)
                data = json.loads(fetch_url(url, timeout=20).decode("utf-8"))
                html = data.get("parse", {}).get("text", {}).get("*", "")
            else:
                url = f"https://{domain}/wiki/{urllib.parse.quote(title)}"
                html = fetch_url(url, timeout=20).decode("utf-8", errors="replace")

            soup = BeautifulSoup(html, "lxml")
            for tag in soup(["script", "style", "nav", "table", "form"]):
                tag.decompose()
            content = soup.find("div", id="mw-content-text") or soup.body
            if content:
                text = content.get_text(separator="\n")
                text = re.sub(r"\n{3,}", "\n\n", text)
                text = re.sub(r"[ \t]+", " ", text).strip()
                if len(text) < 50:
                    continue
                deva = sum(1 for c in text if 0x0900 <= ord(c) <= 0x097F)
                if deva / max(len(text), 1) < 0.15:
                    continue
                rec = {
                    "text": text,
                    "source": "manual",
                    "lang": lang,
                    "dataset": "wikisource",
                    "title": title,
                    "n_chars": len(text),
                    "n_bytes": len(text.encode("utf-8")),
                }
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_out += 1
                total_chars += len(text)
        except:
            pass

        if (i + 1) % 100 == 0:
            print(
                f"  [{i + 1}/{len(titles)}] saved {n_out}, {total_chars / 1e6:.1f} MB",
                flush=True,
            )
        time.sleep(0.3 if use_api else 0.5)

    f_out.close()
    print(f"  Wikisource DONE: {n_out} pages, {total_chars / 1e6:.1f} MB", flush=True)
    return n_out, total_chars


# =====================================================================
# MAIN
# =====================================================================
def main():
    p = argparse.ArgumentParser(description="Comprehensive manual data scraper.")
    p.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--sources",
        default="rss,wikisource,wikipedia,kavitakosh",
        help="Comma-separated: rss,wikisource,wikipedia,kavitakosh",
    )
    p.add_argument(
        "--max-articles", type=int, default=200, help="Max articles per RSS feed."
    )
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    sources = [s.strip() for s in args.sources.split(",")]
    total_n = total_c = 0

    if "rss" in sources:
        print(f"\n=== RSS NEWS FEEDS ({args.lang}) ===", flush=True)
        feeds = HINDI_RSS_FEEDS if args.lang == "hindi" else NEPALI_RSS_FEEDS
        n, c = scrape_rss_feeds(feeds, args.lang, args.output_dir, args.max_articles)
        total_n += n
        total_c += c

    if "wikisource" in sources:
        print(f"\n=== WIKISOURCE ({args.lang}) ===", flush=True)
        domain = "hi.wikisource.org" if args.lang == "hindi" else "ne.wikisource.org"
        use_api = args.lang == "hindi"  # Nepali API is broken
        n, c = scrape_wikisource(domain, args.lang, args.output_dir, use_api=use_api)
        total_n += n
        total_c += c

    if "wikipedia" in sources:
        print(f"\n=== WIKIPEDIA CURATED ({args.lang}) ===", flush=True)
        domain = "hi.wikipedia.org" if args.lang == "hindi" else "ne.wikipedia.org"
        topics = HINDI_WIKI_TOPICS if args.lang == "hindi" else NEPALI_WIKI_TOPICS
        n, c = scrape_wikipedia_articles(domain, topics, args.lang, args.output_dir)
        total_n += n
        total_c += c

    if "kavitakosh" in sources and args.lang == "hindi":
        print(f"\n=== KAVITAKOSH (Hindi literature) ===", flush=True)
        n, c = scrape_kavitakosh(args.lang, args.output_dir)
        total_n += n
        total_c += c

    print(f"\n{'=' * 60}", flush=True)
    print(f"TOTAL ({args.lang}): {total_n} docs, {total_c / 1e6:.1f} MB", flush=True)
    print(f"{'=' * 60}", flush=True)


if __name__ == "__main__":
    main()
