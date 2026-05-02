import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from email_notifier import EmailSettings, _build_digest_html
from html_generator import generate_html_from_json


class PlaceholderTldrTests(unittest.TestCase):
    def test_html_generation_hides_placeholder_tldr_fields(self):
        papers = [
            {
                "title": "Placeholder Paper",
                "summary": "Meaningful English abstract",
                "summary_zh": "Meaningful translated abstract",
                "tldr": "...",
                "tldr_zh": "...",
                "overall_priority_score": 10,
                "authors": ["Author A"],
                "url": "https://example.com/placeholder",
            },
            {
                "title": "Real Paper",
                "summary": "Another abstract",
                "tldr": "A useful TLDR",
                "tldr_zh": "A useful brief",
                "overall_priority_score": 9,
                "authors": ["Author B"],
                "url": "https://example.com/real",
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            json_path = tmp_path / "2026-05-01.json"
            output_dir = tmp_path / "html"
            json_path.write_text(json.dumps(papers), encoding="utf-8")

            generate_html_from_json(
                json_file_path=str(json_path),
                template_dir=str(PROJECT_ROOT / "templates"),
                template_name="paper_template.html",
                output_dir=str(output_dir),
            )

            html = (output_dir / "2026_05_01.html").read_text(encoding="utf-8")

        self.assertIn("Placeholder Paper", html)
        self.assertNotIn("<strong>TLDR</strong>: ...", html)
        self.assertNotIn("<strong>简要说明</strong>: ...", html)
        self.assertIn("<strong>TLDR</strong>: A useful TLDR", html)
        self.assertIn("<strong>简要说明</strong>: A useful brief", html)

    def test_digest_email_falls_back_when_tldr_is_placeholder(self):
        settings = EmailSettings(
            sender="sender@example.com",
            receiver="receiver@example.com",
            smtp_server="smtp.example.com",
            smtp_port=465,
            sender_password="secret",
            pages_base_url="https://example.com",
            max_items=5,
        )
        html = _build_digest_html(
            settings,
            date(2026, 5, 1),
            [
                {
                    "title": "Placeholder Paper",
                    "summary": "Meaningful English abstract",
                    "summary_zh": "Meaningful translated abstract",
                    "tldr": "...",
                    "tldr_zh": "...",
                    "overall_priority_score": 10,
                    "url": "https://example.com/placeholder",
                }
            ],
        )

        self.assertIn("Placeholder Paper", html)
        self.assertNotIn("<p style=\"color:#555;\">...</p>", html)
        self.assertIn("Meaningful translated abstract", html)


if __name__ == "__main__":
    unittest.main()
