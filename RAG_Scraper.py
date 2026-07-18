import time
import requests
import os
import random
from bs4 import BeautifulSoup
from datetime import datetime
from urllib.parse import urlparse, urljoin
import re
import json
import configparser
from html import unescape
import logging
import pymupdf4llm

CURRENT_CTX = {"journal": "", "url": ""}


def log_err(func_name, exc):
    logging.error(f"{CURRENT_CTX.get('journal','')} | {CURRENT_CTX.get('url','')} | {func_name} | {exc}")


def setup_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.ERROR)

    handler = logging.FileHandler(log_file, delay=True)
    handler.setLevel(logging.ERROR)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    logger.addHandler(handler)


def get_base_url(url):
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"


def get_headers():
    headers = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Priority": "u=0, i",
        "Sec-Ch-Ua": '"Chromium";v="134", "Not:A-Brand";v="24", "Google Chrome";v="134"',
        "Sec-Ch-Ua-Arch": '"x86"',
        "Sec-Ch-Ua-Bitness": '"64"',
        "Sec-Ch-Ua-Full-Version": '"134.0.6998.89"',
        "Sec-Ch-Ua-Full-Version-List": '"Chromium";v="134.0.6998.89", "Not:A-Brand";v="24.0.0.0", "Google Chrome";v="134.0.6998.89"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Model": '""',
        "Sec-Ch-Ua-Platform": '"Windows"',
        "Sec-Ch-Ua-Platform-Version": '"15.0.0"',
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36",
    }

    return headers


BLOCK_TITLE_MARKERS = (
    "just a moment",
    "access denied",
    "are you a robot",
    "attention required",
    "captcha",
    "ddos protection",
    "checking your browser",
)

DELAY_BETWEEN_ARTICLES = (2.0, 5.0)   # seconds, random uniform
DELAY_BETWEEN_PAGES = (1.0, 3.0)      # pagination pages within an issue
DELAY_BETWEEN_JOURNALS = (10.0, 20.0)

FOREIGN_ABSTRACT_TITLES = ("Zusammenfassung", "Résumé", "Resumen", "Resumo", "Riassunto")


def create_session():
    # Plain session. All retry logic lives in get_soup — one layer, not two.
    return requests.Session()


def get_soup(url, session, retries=5, base_delay=2.0):
    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, headers=get_headers(), timeout=30)

            if response.status_code == 404:
                logging.error(f"HTTP 404 | {url}")
                return None  # permanent, do not retry

            if response.status_code in (403, 429):
                retry_after = response.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    wait = int(retry_after) + random.uniform(1, 3)
                else:
                    wait = base_delay * (2 ** attempt) + random.uniform(0, 2)
                logging.error(
                    f"BLOCKED status={response.status_code} attempt={attempt}/{retries} "
                    f"wait={wait:.0f}s | {url}"
                )
                time.sleep(wait)
                continue

            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")

            # Sentinel: a 200 can still be a challenge/consent page.
            title_tag = soup.find("title")
            title = title_tag.get_text(strip=True).lower() if title_tag else ""
            if any(marker in title for marker in BLOCK_TITLE_MARKERS):
                logging.error(f"BLOCK PAGE (200) title='{title[:80]}' attempt={attempt}/{retries} | {url}")
                time.sleep(base_delay * (2 ** attempt) + random.uniform(0, 2))
                continue

            return soup

        except requests.RequestException as e:
            logging.error(f"Fetch attempt {attempt}/{retries} failed | {url} | {e}")
            if attempt < retries:
                time.sleep(base_delay * (2 ** (attempt - 1)) + random.uniform(0, 1))

    logging.error(f"FETCH GAVE UP after {retries} attempts | {url}")
    return None


def get_output_path(config_file="config.ini"):
    try:
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"{config_file} not found")

        config = configparser.ConfigParser()
        config.read(config_file)

        output_path = config["DETAILS"]["output_path"]

        now = datetime.now()
        date_folder = now.strftime("%Y%m%d")
        timestamp = now.strftime("%H%M%S")

        date_path = os.path.join(output_path, f"{date_folder}_{timestamp}")

        os.makedirs(date_path, exist_ok=True)

        return date_path

    except (FileNotFoundError, KeyError) as e:
        raise RuntimeError(f"config.ini invalid or missing required key: {e}") from e


def read_urls(file_path="urlDetails.txt"):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            urls = [line.strip() for line in file if line.strip()]

        print(f"✅ Total URLs: {len(urls)}")
        return urls

    except Exception as e:
        raise Exception(f"Error reading urlDetails.txt file: {e}")


def create_json_path(output_folder, journal_title, suffix=""):
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d%H%M%S")

    safe_title = journal_title.replace(" ", "_")

    safe_title = re.sub(r'[<>:"/\\|?*]', "", safe_title)

    safe_title = re.sub(r"[^A-Za-z0-9_-]", "", safe_title)

    if not safe_title:
        safe_title = "untitled"

    journal_folder = os.path.join(output_folder, safe_title)
    os.makedirs(journal_folder, exist_ok=True)

    filename = f"{safe_title}_{timestamp}{suffix}.json"
    json_file = os.path.join(journal_folder, filename)

    return json_file


def extract_journal_title(soup):
    try:
        title_tag = soup.find("a", attrs={"class": "app-journal-masthead__title-link"})
        if title_tag:
            title = title_tag.get_text(strip=True)
            return title
        else:
            return ""
    except Exception as e:
        log_err("extract_journal_title", e)
        return ""


def extract_volume_and_issue(soup):
    try:
        volume_issue_tag = soup.find("h2", class_="app-journal-latest-issue__heading")
        if not volume_issue_tag:
            return "", ""

        text = volume_issue_tag.get_text(strip=True)

        volume_match = re.search(r"Volume\s+(\d+)", text)
        issue_match = re.search(r"Issue\s+(\d+)", text)

        volume = volume_match.group(1) if volume_match else ""
        issue = issue_match.group(1) if issue_match else ""

        return volume, issue

    except Exception as e:
        log_err("extract_volume_and_issue", e)
        return "", ""


def extract_publication_year(soup):
    try:
        time_tag = soup.find("time", class_="app-journal-latest-issue__date")

        if not time_tag:
            return ""

        text = time_tag.get_text(strip=True)

        match = re.search(r"\b(\d{4})\b", text)
        return match.group(1) if match else ""

    except Exception as e:
        log_err("extract_publication_year", e)
        return ""


def get_editors_details(soup):
    main_editor_tag = soup.find("dl", class_="app-journal-latest-issue__editors")
    if not main_editor_tag:
        return []

    try:
        all_edi = main_editor_tag.find("ul", class_="app-journal-latest-issue__editors-list").find_all("li", class_="app-journal-latest-issue__editor")
        edi_list = []
        for sin_edi in all_edi:
            try:
                name = sin_edi.get_text(" ", strip=True)
                name = re.sub(r"\s+", " ", name).strip().rstrip(",")
                edi_list.append(name)
            except Exception as e:
                log_err("get_editors_details", e)

        return edi_list if edi_list else []
    except Exception as e:
        log_err("get_editors_details", e)
        return []


MAX_PAGES_PER_ISSUE = 50


def get_article_content(url, session):
    all_items = []
    seen_hrefs = set()
    page_count = 1

    while page_count <= MAX_PAGES_PER_ISSUE:
        sep = "&" if "?" in url else "?"
        page_url = f"{url}{sep}page={page_count}"
        page_soup = get_soup(page_url, session)

        if page_soup is None:
            logging.error(f"PAGINATION FETCH FAILED | page {page_count} | {page_url}")
            break

        if page_soup.find("div", attrs={"data-test": "no-results"}):
            break
        ol_tag = page_soup.find("ol", attrs={"class": "u-list-reset"})
        if not ol_tag:
            break

        new_on_this_page = 0
        for item in ol_tag.find_all("div", class_="app-card-open__main"):
            link = item.find("h2", class_="app-card-open__heading")
            a_tag = link.find("a") if link else None
            href = a_tag.get("href", "") if a_tag else ""
            if not href or href in seen_hrefs:
                continue
            seen_hrefs.add(href)
            all_items.append(item)
            new_on_this_page += 1

        if new_on_this_page == 0:
            # Page param likely clamped / re-serving same content — stop.
            logging.error(f"PAGINATION STALLED (0 new items) | page {page_count} | {page_url}")
            break

        page_count += 1
        time.sleep(random.uniform(*DELAY_BETWEEN_PAGES))

    if page_count > MAX_PAGES_PER_ISSUE:
        logging.error(f"PAGINATION HIT MAX_PAGES_PER_ISSUE ({MAX_PAGES_PER_ISSUE}) | {url}")

    print(f"✅ Pagination complete: {page_count if page_count <= MAX_PAGES_PER_ISSUE else MAX_PAGES_PER_ISSUE} page(s), {len(all_items)} article(s)")
    return all_items


def extract_article_item_type(soup):
    item_type_tag = soup.find("li", attrs={"data-test": "article-category"})
    if item_type_tag:
        return item_type_tag.get_text(strip=True)
    else:
        return ""


def extract_article_item_date(soup):
    try:
        first_tag = soup.find("ul", attrs={"class": "c-article-identifiers"})
        date_tag = first_tag.find("time", attrs={"datetime": True})

        if not date_tag:
            return "", "", ""

        text = date_tag.get_text(" ", strip=True)

        match = re.search(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", text)

        if match:
            day = match.group(1)
            month = match.group(2)
            year = match.group(3)
            return year, month, day

        return "", "", ""

    except Exception as e:
        log_err("extract_article_item_date", e)
        return "", "", ""


def extract_article_item_page_range(soup):
    try:
        page_range_tag = soup.find("ul", class_="c-article-identifiers c-article-identifiers--cite-list")

        if not page_range_tag:
            return "", ""

        text = page_range_tag.get_text(" ", strip=True)

        match = re.search(r"\bpage(?:s)?\s+(\d+)(?:\s*[–-]\s*(\d+))?", text)

        if match:
            first_page = match.group(1)
            last_page = match.group(2) if match.group(2) else ""
            return first_page, last_page

        return "", ""

    except Exception as e:
        log_err("extract_article_item_page_range", e)
        return "", ""


def extract_article_item_article_id(soup):
    article_number_tag = soup.find("ul", class_="c-article-identifiers c-article-identifiers--cite-list")
    if article_number_tag:
        try:
            article_number = article_number_tag.find("span", attrs={"data-test": "article-number"}).get_text(strip=True)
            return article_number
        except Exception as e:
            log_err("extract_article_item_article_id", e)
            return ""
    else:
        return ""


def extract_article_item_title(soup):
    article_tag = soup.find("article", attrs={"lang": True})
    article_lang = (article_tag.get("lang") or "").lower() if article_tag else ""
    article_section_eng_lang = article_lang.startswith("en") or article_lang == ""
    article_section_non_eng_lang = bool(article_lang) and not article_lang.startswith("en")

    main_div_tag = soup.find("div", class_="app-article-masthead__info")
    if not main_div_tag:
        return "", ""

    foreign_tag = main_div_tag.find("p", class_="c-article-title__sub")

    h1_tag = main_div_tag.find("h1", class_="c-article-title")

    if foreign_tag:
        all_tags_for_eng = main_div_tag.find_all("p", class_="c-article-title__sub", attrs={"lang": "en"})

        eng_title = " ".join(format_title_with_tags(tag) for tag in all_tags_for_eng)

        all_tags_for_foreign = main_div_tag.find_all("p", class_="c-article-title__sub", attrs={"lang": lambda x: x != "en"})

        first_for_title = format_title_with_tags(h1_tag)

        foreign_text = " ".join(format_title_with_tags(tag) for tag in all_tags_for_foreign)

        for_title = f"{first_for_title} {foreign_text}".strip()

        return eng_title, for_title

    else:
        if article_section_eng_lang:
            return format_title_with_tags(h1_tag), ""
        elif article_section_non_eng_lang:
            return "", format_title_with_tags(h1_tag)
        else:
            return format_title_with_tags(h1_tag), ""


def format_title_with_tags(tag):
    if not tag:
        return ""

    for i in tag.find_all(["i", "em"]):
        i.name = "italic"

    for s in tag.find_all("sup"):
        s.name = "sup"

    for s in tag.find_all("sub"):
        s.name = "sub"

    content = tag.decode_contents()
    return unescape(content)


def extract_article_item_doi(soup):
    # 1) Stable meta tags, in preference order
    for meta_name in ("citation_doi", "prism.doi", "dc.identifier", "DOI"):
        tag = soup.find("meta", attrs={"name": meta_name})
        if tag:
            content = (tag.get("content") or "").strip()
            content = content.replace("doi:", "").strip()
            if content.startswith("10."):
                return content

    # 2) Legacy fallback: citeas bibliographic block
    container = extract_bibliographic_block(soup)
    if container:
        doi_abbr = container.find("abbr", title="Digital Object Identifier")
        if doi_abbr:
            li_tag = doi_abbr.find_parent("li")
            if li_tag:
                value_tag = li_tag.find("span", class_="c-bibliographic-information__value")
                if value_tag:
                    text = value_tag.get_text(strip=True)
                    if "doi.org/" in text:
                        return text.split("doi.org/")[-1]

    logging.error("DOI MISSING — record will lack its primary key")
    return ""


def parse(section):
    if not section:
        return ""

    content = section.find("div", class_="c-article-section__content")
    if not content:
        return ""

    result = []
    for tag in content.find_all(["h3", "p"]):
        text = re.sub(r"\s+", " ", tag.get_text(" ", strip=True)).strip()
        if not text:
            continue

        if text.strip().lower() in ["none.", "none", "na", "n/a"]:
            continue
        result.append(text)

    return "\n".join(result)


def extract_article_item_abstract(soup):
    try:
        eng_abs_tag = soup.find("section", attrs={"data-title": "Abstract"})
        for_abs_tag = None
        for title in FOREIGN_ABSTRACT_TITLES:
            for_abs_tag = soup.find("section", attrs={"data-title": title})
            if for_abs_tag:
                break

        eng_sub = parse(eng_abs_tag)
        for_abs = parse(for_abs_tag)

        return eng_sub, for_abs
    except Exception as e:
        log_err("extract_article_item_abstract", e)
        return "", ""


def extract_bibliographic_block(soup):
    cite_tag = soup.find("h3", id="citeas")
    if not cite_tag:
        return None

    parent_div = cite_tag.find_parent("div", class_="c-bibliographic-information__column")
    return parent_div


def extract_article_item_auth_keywords(soup):
    container = extract_bibliographic_block(soup)
    if not container:
        return []

    keyword_tag = container.find("h3", string="Keywords")
    if not keyword_tag:
        return []

    keyword_list_tag = keyword_tag.find_next_sibling("ul", class_="c-article-subject-list")
    if not keyword_list_tag:
        return []

    keywords = []

    for li in keyword_list_tag.find_all("li", class_="c-article-subject-list__subject"):
        text = li.get_text(" ", strip=True)
        if text:
            keywords.append(text)

    return keywords


def extract_article_item_references(soup):
    ref_tag = soup.find("div", attrs={"data-container-section": "references"})
    if not ref_tag:
        return []
    ref_list = ref_tag.find("ol", class_="c-article-references")
    if not ref_list:
        return []

    out = []
    for li in ref_list.find_all("li", class_="c-article-references__item"):
        text_p = li.find("p", class_="c-article-references__text")
        ref_id = text_p.get("id", "") if text_p else ""
        text = re.sub(r"\s+", " ", text_p.get_text(" ", strip=True)).strip() if text_p else ""

        counter = (li.get("data-counter") or "").rstrip(".").strip()
        try:
            index = int(counter)
        except ValueError:
            index = len(out) + 1

        # DOI: prefer data-doi attr; fall back to a clean DOI in an inline link
        doi = None
        doi_anchor = li.find("a", attrs={"data-doi": True})
        if doi_anchor:
            doi = doi_anchor["data-doi"].strip()
        else:
            ext = li.find("a", attrs={"data-track-action": "external reference"})
            if ext and ext.get("data-track-label", "").startswith("10."):
                doi = ext["data-track-label"].strip()

        # PMID / PMCID: extract the ID, never store the URL
        pmid = pmcid = None
        links_p = li.find("p", class_="c-article-references__links")
        if links_p:
            pm = links_p.find("a", attrs={"data-track-value": "pubmed reference"})
            if pm:
                m = re.search(r"list_uids=(\d+)", pm.get("href", ""))
                if m:
                    pmid = m.group(1)
            pmc = links_p.find("a", attrs={"data-track-value": "pubmed central reference"})
            if pmc:
                m = re.search(r"(PMC\d+)", pmc.get("href", ""))
                if m:
                    pmcid = m.group(1)

        out.append(
            {
                "index": index,
                "id": ref_id,
                "text": text,
                "doi": doi,
                "pmid": pmid,
                "pmcid": pmcid,
            }
        )
    return out


def extract_article_item_funding(soup):
    try:
        fund_tag = soup.find("section", attrs={"data-title": "Funding"})

        funding = parse(fund_tag)

        return funding
    except Exception as e:
        log_err("extract_article_item_funding", e)
        return ""


def extract_article_item_license(soup):
    try:
        section_tag = soup.find("section", attrs={"data-title": "Rights and permissions"})
        if not section_tag:
            return ""

        license_tag = section_tag.find("a", attrs={"rel": "license"})
        if not license_tag:
            return ""

        return license_tag.get_text(" ", strip=True)
    except Exception as e:
        log_err("extract_article_item_license", e)
        return ""


def extract_article_item_copy_year(soup):
    try:
        meta_tag = soup.find("meta", attrs={"name": "dc.copyright"})
        if not meta_tag:
            return ""

        content = meta_tag.get("content", "")
        match = re.search(r"\b\d{4}\b", content)

        return match.group(0) if match else ""

    except Exception as e:
        log_err("extract_article_item_copy_year", e)
        return ""


def build_contacts(editors, article_soup):
    article_json = extract_article_json(article_soup)
    authors_json = article_json.get("mainEntity", {}).get("author", [])

    author_names = extract_authors_from_json(article_json)
    if not author_names:
        author_names = extract_article_item_authors(article_soup)

    authors = []
    for name in author_names:
        affiliation, email = get_author_details(name, article_soup, authors_json)
        authors.append({"name": name, "affiliation": affiliation, "email": email})

    corporate_authors = [
        {"name": name, "affiliation": "", "email": ""}
        for name in extract_article_item_corporate(article_soup)
    ]

    editor_objs = [
        {"name": name, "affiliation": "", "email": ""}
        for name in (editors or [])
    ]

    return authors, corporate_authors, editor_objs


def extract_article_item_authors(soup):
    auth_list = []
    section = soup.find("ul", attrs={"data-test": "authors-list"})
    if not section:
        return auth_list

    all_list = section.find_all("li", class_="c-article-author-list__item")

    for sin_auth in all_list:
        try:
            a_tag = sin_auth.find("a", attrs={"data-test": "author-name"})  # specific
            if not a_tag:
                continue
            if a_tag.get("data-author-popup", "").startswith("group"):
                continue
            name = a_tag.get_text(" ", strip=True)
            if name:
                auth_list.append(name)
        except Exception as e:
            log_err("extract_article_item_authors", e)
    return auth_list


def extract_article_item_corporate(soup):
    cop_list = []
    section = soup.find("ul", attrs={"data-test": "authors-list"})
    if not section:
        return cop_list

    all_list = section.find_all("li", class_="c-article-author-list__item")

    for sin_auth in all_list:
        try:
            a_tag = sin_auth.find("a", attrs={"data-test": "author-name"})
            if not a_tag:
                continue

            name = a_tag.get_text(strip=True)

            is_group = "group" in a_tag.get("data-author-popup", "") or "Group" in name or "Working Group" in name or "Consortium" in name or "Team" in name

            if is_group:
                cop_list.append(name)

        except Exception as e:
            log_err("extract_article_item_corporate", e)
    return cop_list


def extract_article_json(soup):
    script_tag = soup.find("script", attrs={"type": "application/ld+json"})
    if not script_tag:
        return {}

    try:
        json_data = json.loads(script_tag.string)
        return json_data
    except Exception as e:
        log_err("extract_article_json", e)
        return {}


def extract_article_affiliations(soup, name):
    try:
        section = soup.find("ol", class_="c-article-author-affiliation__list")
        if not section:
            return ""

        affiliations = []
        aff_list = section.find_all("li")

        for sin_aff in aff_list:
            try:
                auth_name = sin_aff.find("p", class_="c-article-author-affiliation__authors-list")
                if name in auth_name.get_text(" ", strip=True):
                    address = auth_name.find_previous_sibling("p").get_text(" ", strip=True)
                    affiliations.append(address)
            except Exception as e:
                log_err("extract_article_affiliations", e)
        return "\\".join(affiliations)

    except Exception as e:
        log_err("extract_article_affiliations", e)
        return ""


def get_email_by_name(authors, target_name):
    for author in authors:
        if author.get("name") == target_name:
            return author.get("email", "")
    return ""


def get_author_details(name, article_soup, authors):
    if not name:
        return "", ""

    try:
        affiliation = extract_article_affiliations(article_soup, name)
        email = get_email_by_name(authors, name)
        return affiliation, email
    except Exception as e:
        log_err("get_author_details", e)
        return "", ""


def extract_authors_from_json(article_json):
    raw = article_json.get("mainEntity", {}).get("author", [])
    return [a.get("name", "").strip() for a in raw if a.get("@type") == "Person" and a.get("name", "").strip()]


def _clean_text_and_refs(node):
    cited = []
    for a in node.find_all("a", attrs={"data-test": "citation-ref"}):
        m = re.search(r"#(ref-CR\d+)", a.get("href", ""))
        if m:
            cited.append(m.group(1))
    text = node.get_text(" ", strip=True)
    text = re.sub(r"\s*\[[\d,\s–\-]*\]", "", text)  # drop "[16, 17, 18]" markers
    text = re.sub(r"\s+", " ", text).strip()
    seen = set()
    cited = [c for c in cited if not (c in seen or seen.add(c))]
    return text, cited


def extract_article_body(soup):
    container = soup.find("div", class_="main-content") or soup
    SKIP_TITLES = {
        "references",
        "acknowledgements",
        "acknowledgments",
        "author information",
        "funding",
        "rights and permissions",
        "about this article",
        "ethics declarations",
        "notes",
    }
    out = []
    for section in container.find_all("section", attrs={"data-title": True}):
        title = (section.get("data-title") or "").strip()
        if not title or title.lower() in SKIP_TITLES:
            continue
        content = section.find("div", class_="c-article-section__content")
        if content is None:
            continue

        section_obj = {
            "section_index": len(out) + 1,
            "title": title,
            "paragraphs": [],
            "subsections": [],
            "figures": [],
        }
        current = section_obj  # paragraphs land here until a sub-heading opens

        for child in content.find_all(recursive=False):
            classes = child.get("class", [])
            if child.name == "h3" and "c-article__sub-heading" in classes:
                sub = {"heading": child.get_text(" ", strip=True), "paragraphs": []}
                section_obj["subsections"].append(sub)
                current = sub
            elif child.name == "p":
                text, cited = _clean_text_and_refs(child)
                if text:
                    para = {"text": text}
                    if cited:
                        para["cited_refs"] = cited
                    current["paragraphs"].append(para)
            elif child.name == "div" and "c-article-section__figure" in classes:
                cap = child.find("div", class_="c-article-section__figure-description")
                img = child.find("img")
                src = img.get("src", "") if img else ""
                section_obj["figures"].append(
                    {
                        "label": (child.get("data-title") or "").strip(),
                        "caption": _clean_text_and_refs(cap)[0] if cap else "",
                        "image_url": ("https:" + src) if src.startswith("//") else src,
                    }
                )

        out.append(section_obj)
    return out


def extract_article_item_pdf_url(soup):
    if soup is None:
        return None
    a = soup.find("a", attrs={"data-test": "pdf-link"})
    if a is None:
        container = soup.find("div", class_="c-pdf-download")
        a = container.find("a", href=True) if container else None
    href = a.get("href", "").strip() if a else ""
    return href or None


def download_article_pdf(soup, article_url, session, out_json_file, doi):
    href = extract_article_item_pdf_url(soup)
    if not href:
        return "Not Available", ""

    pdf_url = urljoin(get_base_url(article_url), href)
    pdf_dir = os.path.join(os.path.dirname(out_json_file), "pdfs")
    os.makedirs(pdf_dir, exist_ok=True)

    safe = (doi or href.rsplit("/", 1)[-1]).replace("/", "_").replace(".pdf", "")
    pdf_path = os.path.join(pdf_dir, f"{safe}.pdf")

    try:
        resp = session.get(pdf_url, headers=get_headers(), timeout=60)
        resp.raise_for_status()
        if not resp.content[:5].startswith(b"%PDF"):  # magic-byte validation
            logging.error(f"PDF link returned non-PDF content: {pdf_url}")
            return "Not Available", ""
        with open(pdf_path, "wb") as f:
            f.write(resp.content)
        return "Available", pdf_path
    except requests.RequestException as e:
        logging.error(f"PDF download failed {pdf_url}: {e}")
        return "Not Available", ""


def convert_pdf_to_markdown(pdf_path, out_json_file):
    # Sibling "markdown/" folder next to "pdfs/", not nested inside it —
    # keeps raw PDF (Bronze) and derived text (Silver) apart, same split
    # you'll eventually have in S3. Same filename stem as the PDF, same
    # dir-derivation pattern download_article_pdf() already uses for pdf_dir.
    if not pdf_path or not os.path.exists(pdf_path):
        return ""
    md_dir = os.path.join(os.path.dirname(out_json_file), "markdown")
    os.makedirs(md_dir, exist_ok=True)
    stem = os.path.basename(pdf_path).rsplit(".", 1)[0]
    md_path = os.path.join(md_dir, stem + ".md")
    try:
        md_text = pymupdf4llm.to_markdown(pdf_path)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_text)
        return md_path
    except Exception as e:
        logging.error(f"PDF markdown conversion failed {pdf_path}: {e}")
        return ""


def write_json_file(final_data_list, out_json_file, issue_link):
    if not final_data_list:
        print(f"⚠️ No data found. JSON file not created. Current URL: {issue_link}")
        logging.error(f"No data found. JSON file not created. Current URL: {issue_link}")
        return

    tmp = out_json_file + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(final_data_list, f, ensure_ascii=False, indent=2)
    os.replace(tmp, out_json_file)


def data_append_to_list(article, final_data_list):
    final_data_list.append(
        {
            "journal_title": article["journal_title"],
            "journal_url": article["journal_url"],
            "volume": article["volume"],
            "issue": article["issue"],
            "issue_publication_year": article["publication_year"],
            "issue_url": article["issue_url"],
            "article_url": article["url"],
            "item_type": article["item_type"],
            "article_publication_month": article["pub_month"],
            "article_publication_day": article["pub_day"],
            "article_publication_year": article["pub_year"],
            "start_page": article["first_page"],
            "last_page": article["last_page"],
            "article_id": article["article_id"],
            "english_title": article["title"],
            "foreign_title": article["foreign_title"],
            "doi": article["doi"],
            "english_abstract": article["abstract"],
            "foreign_abstract": article["foreign_abstract"],
            "keywords": article["keywords"],
            "authors": article["authors"],
            "corporate_authors": article["corporate_authors"],
            "editors": article["editors"],
            "funding_statement": article["funding"],
            "license_type": article["license"],
            "copyright_year": article["copyright_year"],
            "pdf_status": article["pdf_status"],
            "pdf_path": article["pdf_path"],
            "pdf_markdown_path": article["pdf_markdown_path"],
            "reference_count": len(article["references_list"]),
            "references": article["references_list"],
            "body": article["body"],
        }
    )


def article_process(all_items, issue_details, editors, out_json_file, session):
    total_articles = len(all_items)

    smoke = int(os.environ.get("SMOKE_LIMIT", "30"))
    if smoke > 0:
        all_items = all_items[:smoke]
        total_articles = len(all_items)
        print(f"🔬 SMOKE_LIMIT active: processing only {total_articles} article(s)")

    final_data_list = []

    print(f"✅ Total article count: {total_articles}\n")

    for index, sin_item in enumerate(all_items, start=1):
        article = {}
        try:
            percent = (index / total_articles) * 100
            print(f"⏳ Processing article {index}/{total_articles} ({percent:.1f}%)")

            article_link_tag = sin_item.find("h2", class_="app-card-open__heading")
            article_link_href = article_link_tag.find("a")["href"]
            article_url = urljoin(get_base_url(issue_details["issue_url"]), article_link_href)

            article["url"] = article_url
            CURRENT_CTX["url"] = article_url
            print(f"✅ Article link found: {article['url']}")

            article_soup = get_soup(article_url, session)

            article["journal_title"] = issue_details["journal_title"]
            article["journal_url"] = issue_details["journal_url"]
            article["volume"] = issue_details["volume"]
            article["issue"] = issue_details["issue"]
            article["publication_year"] = issue_details["publication_year"]
            article["issue_url"] = issue_details["issue_url"]

            article["item_type"] = extract_article_item_type(article_soup)
            article["pub_year"], article["pub_month"], article["pub_day"] = extract_article_item_date(article_soup)
            article["first_page"], article["last_page"] = extract_article_item_page_range(article_soup)
            article["article_id"] = extract_article_item_article_id(article_soup)
            article["title"], article["foreign_title"] = extract_article_item_title(article_soup)
            article["doi"] = extract_article_item_doi(article_soup)
            article["abstract"], article["foreign_abstract"] = extract_article_item_abstract(article_soup)
            article["keywords"] = extract_article_item_auth_keywords(article_soup)
            article["references_list"] = extract_article_item_references(article_soup)
            article["body"] = extract_article_body(article_soup)
            article["pdf_status"], article["pdf_path"] = download_article_pdf(article_soup, article["url"], session, out_json_file, article["doi"])
            article["pdf_markdown_path"] = convert_pdf_to_markdown(article["pdf_path"], out_json_file) if article["pdf_status"] == "Available" else ""
            article["funding"] = extract_article_item_funding(article_soup)
            article["license"] = extract_article_item_license(article_soup)
            article["copyright_year"] = extract_article_item_copy_year(article_soup)

            article["authors"], article["corporate_authors"], article["editors"] = build_contacts(editors, article_soup)
            data_append_to_list(article, final_data_list)

            if index % 10 == 0:
                write_json_file(final_data_list, out_json_file, issue_details["issue_url"])

            print(f"✅ Process completed for: {article['title']}\n")
        except Exception as e:
            log_err("article_process", e)
            logging.error(f"Error processing article URL {article.get('url', 'N/A')}: {e}")

        if index < total_articles:
            time.sleep(random.uniform(*DELAY_BETWEEN_ARTICLES))

    write_json_file(final_data_list, out_json_file, issue_details["issue_url"])

    empty_doi = sum(1 for r in final_data_list if not r.get("doi"))
    empty_title = sum(1 for r in final_data_list if not r.get("english_title"))
    empty_abstract = sum(1 for r in final_data_list if not r.get("english_abstract"))
    empty_body = sum(1 for r in final_data_list if not r.get("body"))
    empty_pdf = sum(1 for r in final_data_list if not r.get("pdf_path"))
    empty_authors = sum(1 for r in final_data_list if not r.get("authors"))
    empty_refs = sum(1 for r in final_data_list if not r.get("references"))

    fill_summary = (
        f"FILL SUMMARY | {issue_details.get('journal_title', '')} | records={len(final_data_list)} | "
        f"empty: doi={empty_doi} title={empty_title} abstract={empty_abstract} body={empty_body} "
        f"pdf={empty_pdf} authors={empty_authors} refs={empty_refs}"
    )
    if empty_doi or empty_title:
        logging.error(fill_summary)
    print(fill_summary)

    if final_data_list:
        print(f"✅ Total records: {len(final_data_list)}")
        print(f"✅ JSON file written successfully: {out_json_file}\n")


def process_url(url, output_folder):
    base_url = get_base_url(url)
    session = create_session()
    first_soup = get_soup(url, session)
    issue_details = {}

    all_issue_link_tag = first_soup.find("a", attrs={"data-track": "nav_volumes_and_issues"})
    if all_issue_link_tag:
        all_issue_link = urljoin(base_url, all_issue_link_tag["href"])
    else:
        raise Exception("No link for [View all volumes and issues]")

    second_soup = get_soup(all_issue_link, session)

    issue_details["journal_title"] = extract_journal_title(second_soup)
    issue_details["journal_url"] = all_issue_link
    CURRENT_CTX["journal"] = issue_details["journal_title"]

    journal_title = issue_details["journal_title"]

    if journal_title is None or journal_title == "":
        journal_title = ""

    NUM_ISSUES = int(os.environ.get("NUM_ISSUES", "1"))

    issue_link = second_soup.find("ul", attrs={"data-test": "volumes-and-issues"})

    if issue_link:
        issue_link_tags = issue_link.find_all("li", class_="c-list-group__item")[:NUM_ISSUES]
        if not issue_link_tags:
            raise Exception("No link for current issue")
    else:
        raise Exception("No links for issue")

    total_issues = len(issue_link_tags)
    for issue_num, link_tag in enumerate(issue_link_tags, start=1):
        try:
            current_link_href = link_tag.find("a")["href"]
            current_link = urljoin(base_url, current_link_href)
            print(f"✅ Issue {issue_num}/{total_issues} link found: {current_link}")

            this_issue_details = dict(issue_details)
            this_issue_details["issue_url"] = current_link
            articles_soup = get_soup(current_link, session)
            this_issue_details["volume"], this_issue_details["issue"] = extract_volume_and_issue(articles_soup)
            this_issue_details["publication_year"] = extract_publication_year(articles_soup)

            editors = get_editors_details(articles_soup)

            all_items = get_article_content(current_link, session)

            if not all_items:
                print(f"⚠️ No articles for issue {issue_num}, skipping")
                continue

            print("✅ Found all article links")
            out_json_file = create_json_path(output_folder, journal_title, suffix=f"_issue{issue_num}")
            article_process(all_items, this_issue_details, editors, out_json_file, session)
        except Exception as e:
            log_err("process_url(issue loop)", e)
            logging.error(f"Error processing issue {issue_num}/{total_issues} at {url}: {e}")

        if issue_num < total_issues:
            time.sleep(random.uniform(*DELAY_BETWEEN_ARTICLES))


def main():
    try:
        output_folder = get_output_path()
    except Exception as e:
        print(f"❌ FATAL: cannot start — {e}")
        raise SystemExit(1)

    try:
        print("✅ Config file read successfully")

        log_file = os.path.join(output_folder, "errors.log")
        setup_logger(log_file)
        all_urls = read_urls()
        total_urls = len(all_urls)

        config = configparser.ConfigParser()
        config.read("config.ini")
        root_out = config["DETAILS"]["output_path"]
        completed_file = os.path.join(root_out, f"completed_{datetime.now().strftime('%Y%m%d')}.txt")
        completed = set()
        if os.path.exists(completed_file):
            with open(completed_file, "r", encoding="utf-8") as f:
                completed = {line.strip() for line in f if line.strip()}

        for index, url in enumerate(all_urls, start=1):
            if url in completed:
                print(f"⏭️ Skipping (already completed today): {url}")
                continue

            try:
                print(f"✅ process Scraping started for URL: {url}")
                process_url(url, output_folder)
                with open(completed_file, "a", encoding="utf-8") as f:
                    f.write(url + "\n")
            except Exception as e:
                print(f"⚠️ Error processing URL {url}: {e}")
                logging.error(f"Error processing URL {url}: {e}")

            if index < total_urls:
                time.sleep(random.uniform(*DELAY_BETWEEN_JOURNALS))

    except Exception as e:
        print(f"⚠️ Error in main flow: {e}")
        logging.critical(f"Error in main flow: {e}")


if __name__ == "__main__":
    main()
