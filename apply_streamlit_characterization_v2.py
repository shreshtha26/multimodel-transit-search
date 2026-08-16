from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


IMPORT_LINE = (
    "from streamlit_characterization_v2_panel "
    "import render_characterization_v2_page"
)

REPLACEMENT = '''# =============================================================================
# Page 2 — stellar statistical characterisation v2
# =============================================================================

elif page.startswith("2"):
    render_characterization_v2_page(
        repo_root=REPO_ROOT,
        run_dir=RUN_DIR,
        injections=injections,
        pipelines=pipelines,
        metric_suffix=metric_suffix,
        pipeline_label=pipeline_label,
        header=header,
    )


'''


def add_import(text: str) -> str:
    if IMPORT_LINE in text:
        return text

    anchor = "import streamlit as st\n"
    if anchor not in text:
        raise RuntimeError("Could not find `import streamlit as st` in the target file.")

    return text.replace(anchor, anchor + "\n" + IMPORT_LINE + "\n", 1)


def replace_page_2(text: str) -> str:
    preferred = re.compile(
        r"# =============================================================================\n"
        r"# Page 2 — stars & statistics\n"
        r"# =============================================================================\n\n"
        r"elif page\.startswith\(\"2\"\):.*?"
        r"(?=# =============================================================================\n"
        r"# Page 3 — injection explorer\n"
        r"# =============================================================================)",
        flags=re.S,
    )

    updated, n = preferred.subn(REPLACEMENT, text, count=1)
    if n == 1:
        return updated

    fallback = re.compile(
        r"elif page\.startswith\(\"2\"\):.*?"
        r"(?=elif page\.startswith\(\"3\"\):)",
        flags=re.S,
    )
    updated, n = fallback.subn(
        REPLACEMENT[REPLACEMENT.index('elif page.startswith("2"):'):],
        text,
        count=1,
    )
    if n == 1:
        return updated

    raise RuntimeError(
        "Could not identify the existing Page 2 block. "
        "Use the manual integration snippet instead."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Apply Stellar Characterisation v2 to the Streamlit dashboard."
    )
    parser.add_argument(
        "target",
        type=Path,
        help="Path to the existing Streamlit dashboard .py file.",
    )
    parser.add_argument(
        "--panel",
        type=Path,
        default=Path(__file__).with_name("streamlit_characterization_v2_panel.py"),
        help="Path to streamlit_characterization_v2_panel.py.",
    )
    args = parser.parse_args()

    target = args.target.expanduser().resolve()
    panel = args.panel.expanduser().resolve()

    if not target.exists():
        raise FileNotFoundError(target)
    if not panel.exists():
        raise FileNotFoundError(
            f"Panel module not found: {panel}. "
            "Keep both downloaded files in the same directory or pass --panel."
        )

    original = target.read_text()
    updated = add_import(original)
    updated = replace_page_2(updated)

    backup = target.with_suffix(target.suffix + ".bak_characterization_v2")
    if not backup.exists():
        shutil.copy2(target, backup)

    destination_panel = target.parent / "streamlit_characterization_v2_panel.py"
    if panel != destination_panel:
        shutil.copy2(panel, destination_panel)

    target.write_text(updated)

    print(f"Updated: {target}")
    print(f"Backup:  {backup}")
    print(f"Panel:   {destination_panel}")
    print()
    print("Now run:")
    print(f"  streamlit run {target.name}")


if __name__ == "__main__":
    main()
