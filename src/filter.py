import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _safe_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
        return max(1, value)
    except ValueError:
        logging.warning("Invalid %s=%s, fallback to %s", name, raw, default)
        return default


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_URL = "https://models.sjtu.edu.cn/api/v1/chat/completions"
MODEL_NAME = os.getenv("MODEL_NAME", "minimax-m2.5")


MAX_API_RETRIES = _safe_int_env("OPENROUTER_MAX_RETRIES", 3)
MAX_CONCURRENCY = _safe_int_env("OPENROUTER_MAX_CONCURRENCY", 8)
REQUEST_TIMEOUT_SECONDS = _safe_int_env("OPENROUTER_TIMEOUT_SECONDS", 30)

STRONG_DOMAIN_ANCHORS = (
    "fbg",
    "fiber bragg",
    "optical fiber",
    "surgical",
    "surgery",
    "bronchos",
    "endoscop",
    "catheter",
    "airway",
    "intervention",
    "minimally invasive",
    "soft robot",
    "continuum robot",
)

WEAK_DOMAIN_ANCHORS = (
    "force sensing",
    "shape sensing",
    "shape reconstruction",
    "force estimation",
    "proprioception",
    "navigation",
    "localization",
    "registration",
    "tracking",
    "slam",
    "planning",
    "control",
    "manipulation",
    "robotic",
    "medical",
    "clinical",
    "vla",
    "embodied",
)


def _domain_anchor_decision(title: str, summary: str) -> tuple[bool, str]:
    text = f"{title} {summary}".lower()
    strong_hits = [anchor for anchor in STRONG_DOMAIN_ANCHORS if anchor in text]
    weak_hits = [anchor for anchor in WEAK_DOMAIN_ANCHORS if anchor in text]

    if strong_hits:
        return True, f"strong anchor: {strong_hits[0]}"
    if len(weak_hits) >= 2:
        return True, f"weak anchors: {', '.join(weak_hits[:3])}"
    return False, "insufficient domain anchors"


def _strip_json_fence(text: str) -> str:
    content = text.strip()
    if "```json" in content:
        content = content.split("```json", 1)[1].split("```", 1)[0]
    elif content.startswith("```") and "```" in content[3:]:
        content = content.split("```", 2)[1]
    return content.strip()


def call_openrouter_api(prompt: str, max_tokens: int = 5, retries: int = MAX_API_RETRIES) -> Optional[str]:
    """Call the model API with retry support."""
    if not OPENROUTER_API_KEY:
        logging.error("OPENROUTER_API_KEY is not set. Cannot call model API.")
        return None

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }

    for attempt in range(1, retries + 1):
        try:
            response = requests.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=data,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"].strip()
            if not content:
                raise ValueError("Empty model response")
            return content
        except (requests.exceptions.RequestException, KeyError, IndexError, ValueError) as e:
            logging.warning(
                "Model API failed (attempt %s/%s): %s",
                attempt,
                retries,
                e,
            )
            if attempt < retries:
                # Exponential backoff: 1s, 2s, 4s
                time.sleep(2 ** (attempt - 1))
            else:
                logging.error("Model API failed after %s retries.", retries)
                return None
        except Exception as e:
            logging.error("Unexpected model API error: %s", e, exc_info=True)
            if attempt < retries:
                time.sleep(2 ** (attempt - 1))
            else:
                return None


def filter_papers_by_topic(
    papers: list,
    topic: str = (
        "FBG sensing, FBG force sensing algorithms, FBG shape sensing algorithms, "
        "surgical robotics, surgical robot navigation, bronchoscopy navigation algorithms, "
        "soft robotics, and vision-language-action methods for sensing, estimation, "
        "planning, and control in these domains"
    ),
) -> list:
    """Filter papers by topic relevance using concurrent model calls."""
    if not OPENROUTER_API_KEY:
        logging.error("OPENROUTER_API_KEY is not set. Skip filtering and return original papers.")
        return papers

    total = len(papers)
    if total == 0:
        return papers

    logging.info("Start filtering %s papers with concurrency=%s", total, MAX_CONCURRENCY)

    def _filter_one(i: int, paper: dict):
        title = paper.get("title", "N/A")
        summary = paper.get("summary", "N/A")
        allow_by_anchor, anchor_reason = _domain_anchor_decision(title, summary)
        if not allow_by_anchor:
            logging.info(
                "Paper %s/%s skipped by domain-anchor gate (%s): %s",
                i + 1,
                total,
                anchor_reason,
                title[:100],
            )
            return i, paper, False
        logging.info(
            "Paper %s/%s passed domain-anchor gate (%s): %s",
            i + 1,
            total,
            anchor_reason,
            title[:100],
        )
        prompt = (
            "You are selecting papers for FBG-driven surgical robotics and navigation research. "
            "Answer with ONLY 'yes' or 'no'. "
            "Say 'yes' ONLY if the main contribution is an algorithm/model that is explicitly grounded in at least one target domain: "
            "FBG sensing, FBG force sensing, FBG shape sensing, surgical robotics, surgical robot navigation, "
            "bronchoscopy, endoscopy, catheter navigation, or soft robotics; include VLA/VLM/LLM methods only when they are clearly "
            "applied to sensing, estimation, planning, control, localization, registration, SLAM, tracking, or guidance "
            "in these domains. "
            "Generic VLA foundation-model papers, mechanistic interpretability papers, and pure benchmark/scaling/efficiency papers "
            "without explicit grounding in these domains must be answered 'no'. "
            "Say 'no' for purely hardware/material/fabrication papers without algorithmic contribution, "
            "pure optics/physics theory without sensing or robotics/medical algorithm use, "
            "general NLP/CV/LLM papers without clear domain linkage, or purely clinical workflow reports without methods. "
            f"\nTitle: {title}\nAbstract: {summary}"
        )

        response = call_openrouter_api(prompt, max_tokens=5, retries=MAX_API_RETRIES)
        keep = response is not None and "yes" in response.lower()

        if response is None:
            logging.warning("Paper %s/%s filter failed after retries: %s", i + 1, total, title[:60])
        else:
            logging.info("Paper %s/%s filter response: %s", i + 1, total, response[:100])

        return i, paper, keep

    workers = min(MAX_CONCURRENCY, max(1, total))
    kept = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_filter_one, i, paper) for i, paper in enumerate(papers)]
        for future in as_completed(futures):
            i, paper, keep = future.result()
            if keep:
                kept.append((i, paper))

    kept.sort(key=lambda x: x[0])
    filtered = [paper for _, paper in kept]
    logging.info("Filtering finished: %s/%s papers kept.", len(filtered), total)
    return filtered


rating_prompt_template = """
# Role Setting
You are an experienced researcher in the field of Artificial Intelligence, skilled at quickly evaluating the potential value of research papers.

# Task
Based on the following paper's title and abstract, please summarize it and score it across multiple dimensions (1-10 points, 1 being the lowest, 10 being the highest). Finally, provide an overall preliminary priority score.

# Input
Paper Title: %s
Paper Abstract: %s

# My Research Interests
FBG Sensing + FBG Force/Shape Sensing Algorithms + Surgical Robotics + Surgical Robot Navigation + Bronchoscopy Navigation Algorithms + Soft Robotics + Vision-Language-Action (VLA) for Sensing, Estimation, Planning, and Control, but only when explicitly applied to these domains

# Output Requirements
Output should always be in JSON format, strictly compliant with RFC8259.
Please output the evaluation and explanations in the following JSON format:
{
  "tldr": "<summary>", // Too Long; Didn't Read. Summarize the paper in one or two brief sentences.
  "tldr_zh": "<summary>", // Too Long; Didn't Read. Summarize the paper in one or two brief sentences, in Chinese.
  "relevance_score": <score>, // Relevance to my research interests
  "novelty_claim_score": <score>, // Degree of novelty claimed in the abstract
  "clarity_score": <score>, // Clarity and completeness of the abstract writing
  "potential_impact_score": <score>, // Estimated potential impact based on abstract claims
  "overall_priority_score": <score> // Preliminary reading priority score combining all factors above
}

# Scoring Guidelines
- Relevance: Score high only when the paper is explicitly grounded in FBG, surgical robotics, bronchoscopy/endoscopy/catheter navigation, or soft robotics. Generic VLA papers without this grounding should score low on relevance.
- Novelty: Evaluate the degree of innovation claimed in the abstract regarding the method or viewpoint compared to known work.
- Clarity: Evaluate whether the abstract itself is easy to understand and complete with essential elements.
- Potential Impact: Evaluate the importance of the problem it claims to solve and the potential application value of the results.
- Overall Priority: Provide an overall score combining all the above factors. A high score indicates suggested priority for reading.
"""


def rate_papers(papers: list) -> list:
    """Rate papers concurrently, with up to 3 retries on failures."""
    if not OPENROUTER_API_KEY:
        logging.error("OPENROUTER_API_KEY is not set. Skip rating and return original papers.")
        return papers

    total = len(papers)
    if total == 0:
        return papers

    logging.info("Start rating %s papers with concurrency=%s", total, MAX_CONCURRENCY)

    def _rate_one(i: int, paper: dict):
        title = paper.get("title", "N/A")
        summary = paper.get("summary", "N/A")
        prompt = rating_prompt_template % (title, summary)
        out = dict(paper)

        for attempt in range(1, MAX_API_RETRIES + 1):
            response = call_openrouter_api(prompt, max_tokens=1000, retries=MAX_API_RETRIES)
            if not response:
                continue
            try:
                rating = json.loads(_strip_json_fence(response))
                out.update(rating)
                logging.info("Paper %s/%s rating success (attempt %s)", i + 1, total, attempt)
                return i, out
            except Exception as e:
                logging.warning(
                    "Paper %s/%s rating parse failed (attempt %s/%s): %s",
                    i + 1,
                    total,
                    attempt,
                    MAX_API_RETRIES,
                    e,
                )

        logging.error("Paper %s/%s rating failed after %s attempts.", i + 1, total, MAX_API_RETRIES)
        return i, out

    workers = min(MAX_CONCURRENCY, max(1, total))
    results = [None] * total
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_rate_one, i, paper) for i, paper in enumerate(papers)]
        for future in as_completed(futures):
            i, rated_paper = future.result()
            results[i] = rated_paper

    return [paper for paper in results if paper is not None]


def translate_summaries(papers: list, target_language: str = "中文") -> list:
    """Translate summaries concurrently, with up to 3 retries on failures."""
    if not OPENROUTER_API_KEY:
        logging.error("OPENROUTER_API_KEY is not set. Skip translation and return original papers.")
        return papers

    total = len(papers)
    if total == 0:
        return papers

    logging.info("Start translating %s papers with concurrency=%s", total, MAX_CONCURRENCY)

    def _translate_one(i: int, paper: dict):
        out = dict(paper)
        summary = out.get("summary", "")
        if not summary or summary == "N/A":
            return i, out

        prompt = (
            f"请将以下英文论文摘要翻译成{target_language}。"
            "要求：保持专业术语准确，翻译流畅自然，保留原文技术含义。"
            "只输出翻译结果，不要添加解释。"
            f"\n\n摘要：\n{summary}"
        )

        for attempt in range(1, MAX_API_RETRIES + 1):
            response = call_openrouter_api(prompt, max_tokens=2000, retries=MAX_API_RETRIES)
            if response and response.strip():
                content = response.strip()
                if content.startswith("```") and "```" in content[3:]:
                    content = content.split("```", 2)[1].strip()
                    if "\n" in content and (
                        content.lower().startswith("text") or content.lower().startswith("markdown")
                    ):
                        content = content.split("\n", 1)[1]
                out["summary_zh"] = content.strip().strip('"').strip("'").strip()
                logging.info("Paper %s/%s translation success (attempt %s)", i + 1, total, attempt)
                return i, out

            logging.warning(
                "Paper %s/%s translation failed (attempt %s/%s)",
                i + 1,
                total,
                attempt,
                MAX_API_RETRIES,
            )

        logging.error("Paper %s/%s translation failed after %s attempts.", i + 1, total, MAX_API_RETRIES)
        return i, out

    workers = min(MAX_CONCURRENCY, max(1, total))
    results = [None] * total
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_translate_one, i, paper) for i, paper in enumerate(papers)]
        for future in as_completed(futures):
            i, translated_paper = future.result()
            results[i] = translated_paper

    return [paper for paper in results if paper is not None]
