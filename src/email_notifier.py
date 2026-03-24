import json
import logging
import os
import smtplib
from dataclasses import dataclass
from datetime import date, datetime
from email.header import Header
from email.mime.text import MIMEText
from email.utils import formataddr, parseaddr
from html import escape
from typing import Optional

DEFAULT_PAGES_BASE_URL = ""


def _as_bool(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class EmailSettings:
    sender: str
    receiver: str
    smtp_server: str
    smtp_port: int
    sender_password: str
    sender_name: str = "GitHub Action"
    subject_prefix: str = "ArXiv Daily"
    pages_base_url: str = ""
    send_empty: bool = False
    max_items: int = 20


def load_email_settings_from_env() -> Optional[EmailSettings]:
    sender = os.getenv("EMAIL_SENDER", "").strip()
    receiver = os.getenv("EMAIL_RECEIVER", "").strip()
    smtp_server = os.getenv("EMAIL_SMTP_SERVER", "").strip()
    sender_password = os.getenv("EMAIL_SENDER_PASSWORD", "").strip()
    smtp_port_raw = os.getenv("EMAIL_SMTP_PORT", "465").strip()

    required_missing = [
        key
        for key, value in {
            "EMAIL_SENDER": sender,
            "EMAIL_RECEIVER": receiver,
            "EMAIL_SMTP_SERVER": smtp_server,
            "EMAIL_SENDER_PASSWORD": sender_password,
        }.items()
        if not value
    ]
    if required_missing:
        logging.info(
            "Email notifier disabled (missing env): %s",
            ", ".join(required_missing),
        )
        return None

    try:
        smtp_port = int(smtp_port_raw)
    except ValueError:
        logging.warning("Invalid EMAIL_SMTP_PORT='%s', fallback to 465.", smtp_port_raw)
        smtp_port = 465

    max_items_raw = os.getenv("EMAIL_MAX_ITEMS", "20").strip()
    try:
        max_items = max(1, int(max_items_raw))
    except ValueError:
        logging.warning("Invalid EMAIL_MAX_ITEMS='%s', fallback to 20.", max_items_raw)
        max_items = 20

    return EmailSettings(
        sender=sender,
        receiver=receiver,
        smtp_server=smtp_server,
        smtp_port=smtp_port,
        sender_password=sender_password,
        sender_name=os.getenv("EMAIL_SENDER_NAME", "GitHub Action").strip() or "GitHub Action",
        subject_prefix=os.getenv("EMAIL_SUBJECT_PREFIX", "ArXiv Daily").strip() or "ArXiv Daily",
        pages_base_url=os.getenv("PAGES_BASE_URL", DEFAULT_PAGES_BASE_URL).strip().rstrip("/"),
        send_empty=_as_bool(os.getenv("EMAIL_SEND_EMPTY"), default=True),
        max_items=max_items,
    )


def _format_addr(raw: str) -> str:
    name, addr = parseaddr(raw)
    return formataddr((Header(name, "utf-8").encode(), addr))


def _build_report_url(settings: EmailSettings, target_date: date) -> str:
    if not settings.pages_base_url:
        return ""
    filename = target_date.strftime("%Y_%m_%d") + ".html"
    return f"{settings.pages_base_url}/daily_html/{filename}"


def _build_digest_html(settings: EmailSettings, target_date: date, papers: list[dict]) -> str:
    date_str = target_date.isoformat()
    report_url = _build_report_url(settings, target_date)
    title = f"{escape(settings.subject_prefix)} - {date_str}"
    body_lines = [
        "<html><body>",
        f"<h2>{title}</h2>",
        f"<p>Found <strong>{len(papers)}</strong> papers for {escape(date_str)}.</p>",
    ]
    if settings.pages_base_url:
        home_url = settings.pages_base_url + "/"
        body_lines.append(f'<p>Website: <a href="{escape(home_url)}">{escape(home_url)}</a></p>')

    if report_url:
        body_lines.append(f'<p>Daily report: <a href="{escape(report_url)}">{escape(report_url)}</a></p>')

    if papers:
        body_lines.append("<ol>")
        for paper in papers[: settings.max_items]:
            paper_title = escape(paper.get("title", "Untitled"))
            paper_url = escape(paper.get("url", ""))
            score = paper.get("overall_priority_score", "-")
            tldr = paper.get("tldr_zh") or paper.get("tldr") or paper.get("summary_zh") or paper.get("summary") or ""
            tldr = escape(tldr)
            if paper_url:
                body_lines.append(
                    f'<li><p><a href="{paper_url}">{paper_title}</a> '
                    f'<span style="color:#666;">(score: {escape(str(score))})</span></p>'
                    f'<p style="color:#555;">{tldr}</p></li>'
                )
            else:
                body_lines.append(
                    f"<li><p>{paper_title} "
                    f'<span style="color:#666;">(score: {escape(str(score))})</span></p>'
                    f'<p style="color:#555;">{tldr}</p></li>'
                )
        body_lines.append("</ol>")
    else:
        body_lines.append("<p>No papers passed filtering today.</p>")

    body_lines.append("</body></html>")
    return "\n".join(body_lines)


def _send_html_email(settings: EmailSettings, subject: str, html: str):
    msg = MIMEText(html, "html", "utf-8")
    msg["From"] = _format_addr(f"{settings.sender_name} <{settings.sender}>")
    bcc_receivers = [x.strip() for x in settings.receiver.split(",") if x.strip()]
    if not bcc_receivers:
        raise ValueError("No valid receiver found in EMAIL_RECEIVER.")

    # Do not expose recipient list in header; actual recipients are passed to sendmail().
    msg["To"] = _format_addr(f"Undisclosed Recipients <{settings.sender}>")
    msg["Subject"] = Header(subject, "utf-8").encode()

    server = None
    try:
        server = smtplib.SMTP(settings.smtp_server, settings.smtp_port, timeout=20)
        server.starttls()
        logging.info("Email transport: SMTP + STARTTLS")
    except Exception as tls_error:
        logging.info("STARTTLS failed (%s). Trying SMTP_SSL...", tls_error)
        try:
            server = smtplib.SMTP_SSL(settings.smtp_server, settings.smtp_port, timeout=20)
            logging.info("Email transport: SMTP_SSL")
        except Exception as ssl_error:
            logging.info("SMTP_SSL failed (%s). Falling back to plain SMTP...", ssl_error)
            server = smtplib.SMTP(settings.smtp_server, settings.smtp_port, timeout=20)
            logging.info("Email transport: plain SMTP")

    try:
        server.login(settings.sender, settings.sender_password)
        server.sendmail(settings.sender, bcc_receivers, msg.as_string())
        logging.info("Email sent via Bcc to %s recipients.", len(bcc_receivers))
    finally:
        server.quit()


def send_daily_digest_if_configured(target_date: date, json_file_path: str) -> bool:
    settings = load_email_settings_from_env()
    if settings is None:
        return False

    if not os.path.exists(json_file_path):
        logging.warning("Skip email: JSON file not found at %s", json_file_path)
        return False

    try:
        with open(json_file_path, "r", encoding="utf-8") as f:
            papers = json.load(f)
    except Exception as e:
        logging.error("Skip email: failed reading JSON %s (%s)", json_file_path, e)
        return False

    if not papers and not settings.send_empty:
        logging.info("No papers and EMAIL_SEND_EMPTY=false. Skip email.")
        return False

    subject_date = datetime.now().strftime("%Y/%m/%d")
    subject = f"{settings.subject_prefix} {subject_date}"
    html = _build_digest_html(settings, target_date, papers)
    try:
        _send_html_email(settings, subject, html)
        logging.info("Digest email sent to %s", settings.receiver)
        return True
    except Exception as e:
        logging.error("Failed to send digest email: %s", e, exc_info=True)
        return False
