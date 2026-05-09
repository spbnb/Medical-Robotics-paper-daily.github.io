import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from filter import _domain_anchor_decision, filter_papers_by_topic, rating_prompt_template
from main import FETCH_CATEGORIES


class TopicConfigTests(unittest.TestCase):
    def test_fetch_categories_include_multiagent_systems(self):
        self.assertIn("cs.MA", FETCH_CATEGORIES)

    def test_medical_agents_are_domain_anchors(self):
        allowed, reason = _domain_anchor_decision(
            "Medical AI Agents for Clinical Decision Support",
            "We propose an LLM-based medical agent for patient triage and care planning.",
        )

        self.assertTrue(allowed)
        self.assertIn("medical agent", reason)

    def test_requested_intervention_and_copilot_topics_are_domain_anchors(self):
        examples = [
            (
                "surgical copilot",
                "A Surgical Copilot for Robot-Assisted Procedure Planning",
                "The system supports intraoperative decision making with an LLM-based assistant.",
            ),
            (
                "clinical copilot",
                "Clinical Copilot Agents for Perioperative Care",
                "We develop a multi-agent workflow for patient-specific clinical recommendations.",
            ),
            (
                "image-guided intervention",
                "Image-Guided Intervention Planning with Foundation Models",
                "The method localizes targets and plans instrument trajectories from medical images.",
            ),
            (
                "computer-assisted intervention",
                "Computer-Assisted Intervention for Endovascular Navigation",
                "The algorithm estimates vessel centerlines and recommends navigation actions.",
            ),
            (
                "computer-assisted surgery",
                "Computer-Assisted Surgery with Multimodal Robot Guidance",
                "The model fuses surgical video and tracking data for intraoperative guidance.",
            ),
        ]

        for expected_reason, title, summary in examples:
            with self.subTest(expected_reason=expected_reason):
                allowed, reason = _domain_anchor_decision(title, summary)
                self.assertTrue(allowed)
                self.assertIn(expected_reason, reason)

    def test_prompts_include_medical_agent_interest(self):
        self.assertIn("medical AI agents", filter_papers_by_topic.__defaults__[0])
        self.assertIn("medical AI agents", rating_prompt_template)
        for phrase in (
            "surgical copilots",
            "clinical copilots",
            "image-guided intervention",
            "computer-assisted intervention",
            "computer-assisted surgery",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, filter_papers_by_topic.__defaults__[0])
                self.assertIn(phrase, rating_prompt_template)


if __name__ == "__main__":
    unittest.main()
