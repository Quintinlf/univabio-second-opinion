"""Screenshot the running app, one image per beat of the demo video.

    streamlit run app.py            # in one terminal
    python scripts/capture_cue_cards.py --port 8501

Writes PNGs to submission/cue_cards/. These are the cue cards in
submission/VIDEO_SCRIPT.md: each one is the frame that should be on screen
while the matching paragraph is read aloud.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "submission" / "cue_cards"

PRESETS = [
    ("Typical benign nucleus", "03_benign_decisive"),
    ("Typical malignant nucleus", "04_malignant_decisive"),
    ("Borderline (withhold)", "05_abstain_with_next_step"),
    ("Implausible geometry", "06_out_of_distribution"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8501)
    ap.add_argument("--width", type=int, default=1500)
    ap.add_argument("--height", type=int, default=1000)
    args = ap.parse_args()

    from playwright.sync_api import sync_playwright

    OUT.mkdir(parents=True, exist_ok=True)
    url = f"http://localhost:{args.port}"

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        page = browser.new_page(viewport={"width": args.width, "height": args.height},
                                device_scale_factor=2)
        page.goto(url, wait_until="networkidle", timeout=120_000)
        # First load fits the model and builds the presets.
        page.wait_for_selector("text=Assessment", timeout=180_000)
        page.wait_for_timeout(2500)

        def shot(name: str, full: bool = False) -> None:
            page.screenshot(path=str(OUT / f"{name}.png"), full_page=full)
            print(f"  wrote {name}.png")

        print("capturing:")
        shot("01_hero")

        # Glossary open — the answer to "what do these words mean".
        glossary = page.locator("summary", has_text="Plain-English glossary").first
        glossary.click()
        page.wait_for_timeout(1500)
        shot("02_glossary")
        glossary.click()
        page.wait_for_timeout(900)

        for label, name in PRESETS:
            box = page.get_by_role("combobox", name="Cell profile")
            box.click()
            page.wait_for_timeout(600)
            page.get_by_role("option", name=label, exact=True).click()
            page.wait_for_timeout(3000)
            shot(name)

        # Evidence section: scroll so the two calibration charts fill the frame.
        page.get_by_text("Why you should believe any of this").first.scroll_into_view_if_needed()
        page.wait_for_timeout(1800)
        shot("07_calibration_charts")

        page.get_by_text("The headline numbers", exact=False).first.scroll_into_view_if_needed()
        page.wait_for_timeout(1500)
        shot("08_metrics_with_bands")

        page.get_by_text("here are all 40", exact=False).first.scroll_into_view_if_needed()
        page.wait_for_timeout(1500)
        shot("09_forty_splits")

        noise = page.locator("summary", has_text="What happens under measurement noise").first
        noise.scroll_into_view_if_needed()
        noise.click()
        page.wait_for_timeout(3000)
        noise.scroll_into_view_if_needed()
        page.wait_for_timeout(1500)
        shot("10_shift_charts")

        shot("11_full_page", full=True)
        browser.close()

    print(f"\n{len(list(OUT.glob('*.png')))} cue cards in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
