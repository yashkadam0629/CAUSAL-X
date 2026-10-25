"""Capture README screenshots from a running dashboard (needs: pip install playwright; uses the installed Edge/Chrome).

    python scripts/capture_screenshots.py [http://127.0.0.1:5173]
Requires a trained dataset (see README) and both servers running.
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:5173"
OUT = Path(__file__).resolve().parents[1] / "docs" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)


def click(page, text, timeout=15000):
    page.get_by_role("button", name=text).first.click(timeout=timeout)


with sync_playwright() as p:
    try:
        browser = p.chromium.launch(channel="msedge")
    except Exception:
        browser = p.chromium.launch(channel="chrome")
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    page.set_default_timeout(120000)

    def shot(name, full=True):
        page.wait_for_timeout(800)
        page.screenshot(path=str(OUT / f"{name}.png"), full_page=full)
        print("saved", name)

    def go(path, wait_text):
        page.goto(BASE + path)
        page.get_by_text(wait_text).first.wait_for()

    go("/", "Optimised policy vs highest-risk-first"); shot("01-command-center")
    go("/portfolio", "Portfolio Explorer"); page.locator("tbody tr").first.wait_for(); page.locator("tbody tr").first.click()
    page.get_by_text("Counterfactual explorer").wait_for(); shot("03-decision-explanation", full=False)
    page.keyboard.press("Escape"); page.get_by_label("Close").click(); shot("02-portfolio")
    go("/causal", "Causal question"); click(page, "Run analysis"); page.get_by_text("Estimator comparison").wait_for()
    click(page, "Run sensitivity analysis"); page.get_by_text("Placebo outcome").first.wait_for(); shot("04-causal-lab")
    go("/uplift", "Treatment-effect calibration"); shot("05-uplift")
    go("/optimizer", "Constraints & assumptions"); click(page, "Run optimisation"); page.get_by_text("Strategy comparison").first.wait_for(); shot("06-optimizer")
    go("/scenarios", "Scenarios"); click(page, "Run scenarios"); page.get_by_text("Scenario table").wait_for()
    click(page, "Run advanced analyses"); page.get_by_text("Value of information:").wait_for(); shot("07-scenario-studio")
    go("/policy", "Policy Evaluation"); click(page, "Evaluate all policies"); page.get_by_text("Policy table").wait_for(); shot("08-policy-evaluation")
    go("/experiments", "Design inputs"); click(page, "Generate protocol"); page.get_by_text("Draft protocol").wait_for(); shot("09-experiment-designer")
    go("/models", "Model comparison"); shot("10-model-lab")
    go("/data", "Generate synthetic portfolio"); shot("11-data-quality")
    go("/governance", "Recommendation history"); shot("12-governance")
    page.get_by_role("tab", name="Monitoring & drift").click(); page.get_by_text("Feature drift").first.wait_for(); shot("13-monitoring")
    browser.close()
