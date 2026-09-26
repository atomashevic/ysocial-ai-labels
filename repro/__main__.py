"""Rebuild every manuscript figure and table: ``python -m repro``."""

from __future__ import annotations

import shutil
import sys

from repro import (fig1_study_design, fig2_reply_target_choice, fig3_direct_thread_entry,
                   fig4_prior_contact, figS3_daily_activity, tables)


def main() -> int:
    fig1_args = []
    if shutil.which("rsvg-convert") is None:
        print("rsvg-convert not found: writing Fig. 1 as SVG only", file=sys.stderr)
        fig1_args = ["--svg-only"]
    fig1_study_design.main(fig1_args)
    fig2_reply_target_choice.main([])
    fig3_direct_thread_entry.main([])
    fig4_prior_contact.main([])  # also Supplementary Figs. S1 and S2
    figS3_daily_activity.main([])
    tables.main([])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
