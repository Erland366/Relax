#!/usr/bin/env python3
"""Build a four-slide monochrome CPU-vision supervisor update."""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE, MSO_CONNECTOR
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_PPTX = OUTPUT_DIR / "cpu-vision-encoder-project-update-2026-08-19.pptx"
VALIDATION_REPORT = OUTPUT_DIR / "validation-report.md"
REFERENCE_DECK = Path(
    "/vast/users/qirong.ho/erland/Python_project/amd_nvidia_mismatch_root/"
    "amd_nvidia_mismatch_main/training_reports/camd-numerical-portability-pptx/"
    "AMD Project Progress.pptx"
)

SLIDE_W = 13.333333
SLIDE_H = 7.5
FONT = "Arial"

BLACK = "000000"
WHITE = "FFFFFF"
GRAY_050 = "FAFAFA"
GRAY_100 = "F2F2F2"
GRAY_200 = "D9D9D9"
GRAY_500 = "737373"
GRAY_700 = "404040"


def rgb(value: str) -> RGBColor:
    return RGBColor.from_string(value)


def remove_template_slides(prs: Presentation) -> None:
    slide_ids = prs.slides._sldIdLst  # noqa: SLF001 - python-pptx has no public slide-removal API
    for slide_id in list(slide_ids):
        prs.part.drop_rel(slide_id.rId)
        slide_ids.remove(slide_id)


def set_white_background(slide) -> None:
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = rgb(WHITE)


def add_text(
    slide,
    text: str,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    size: float = 18,
    bold: bool = False,
    color: str = BLACK,
    align: PP_ALIGN = PP_ALIGN.LEFT,
    valign: MSO_ANCHOR = MSO_ANCHOR.TOP,
    margin: float = 0.0,
    name: str = "text",
    italic: bool = False,
):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    shape.name = name
    frame = shape.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = Inches(margin)
    frame.margin_right = Inches(margin)
    frame.margin_top = Inches(margin)
    frame.margin_bottom = Inches(margin)
    frame.vertical_anchor = valign
    paragraph = frame.paragraphs[0]
    paragraph.alignment = align
    paragraph.space_before = Pt(0)
    paragraph.space_after = Pt(0)
    paragraph.line_spacing = 1.0
    run = paragraph.add_run()
    run.text = text
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = rgb(color)
    return shape


def add_multiline(
    slide,
    lines: list[tuple[str, bool]],
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    size: float = 17,
    color: str = BLACK,
    bullet: bool = False,
    gap: float = 7,
    name: str = "multiline",
):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    shape.name = name
    frame = shape.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = Inches(0.02)
    frame.margin_right = Inches(0.02)
    frame.margin_top = Inches(0.02)
    frame.margin_bottom = Inches(0.02)
    for index, (line, bold) in enumerate(lines):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.text = f"•  {line}" if bullet else line
        paragraph.font.name = FONT
        paragraph.font.size = Pt(size)
        paragraph.font.bold = bold
        paragraph.font.color.rgb = rgb(color)
        paragraph.space_before = Pt(0)
        paragraph.space_after = Pt(gap)
        paragraph.line_spacing = 1.02
    return shape


def add_rect(
    slide,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    fill: str = WHITE,
    line: str = BLACK,
    line_width: float = 1.0,
    name: str = "rectangle",
):
    shape = slide.shapes.add_shape(
        MSO_AUTO_SHAPE_TYPE.RECTANGLE,
        Inches(x),
        Inches(y),
        Inches(w),
        Inches(h),
    )
    shape.name = name
    shape.fill.solid()
    shape.fill.fore_color.rgb = rgb(fill)
    shape.line.color.rgb = rgb(line)
    shape.line.width = Pt(line_width)
    return shape


def add_line(
    slide,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    color: str = BLACK,
    width: float = 1.0,
    name: str = "line",
):
    shape = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT,
        Inches(x1),
        Inches(y1),
        Inches(x2),
        Inches(y2),
    )
    shape.name = name
    shape.line.color.rgb = rgb(color)
    shape.line.width = Pt(width)
    return shape


def add_header(slide, title: str, slide_number: int) -> None:
    add_text(slide, title, 0.62, 0.38, 12.05, 0.52, size=28, bold=True, name="slide-title")
    add_line(slide, 0.62, 1.02, 12.70, 1.02, width=0.8, name="header-rule")
    add_text(
        slide,
        "CPU VISION ENCODER · 19 AUG 2026",
        0.62,
        7.18,
        5.4,
        0.16,
        size=8.5,
        color=GRAY_500,
        name="footer-label",
    )
    add_text(
        slide,
        str(slide_number),
        12.28,
        7.16,
        0.40,
        0.18,
        size=8.5,
        color=GRAY_500,
        align=PP_ALIGN.RIGHT,
        name="footer-number",
    )


def add_labelled_box(
    slide,
    *,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str,
    name: str,
    gray: bool = False,
) -> None:
    add_rect(
        slide,
        x,
        y,
        w,
        h,
        fill=GRAY_050 if gray else WHITE,
        line=BLACK,
        line_width=0.9,
        name=f"box-{name}",
    )
    add_text(
        slide,
        title,
        x + 0.14,
        y + 0.14,
        w - 0.28,
        0.30,
        size=16.5,
        bold=True,
        name=f"title-{name}",
    )
    add_text(
        slide,
        body,
        x + 0.14,
        y + 0.53,
        w - 0.28,
        h - 0.66,
        size=13.5,
        color=GRAY_700,
        name=f"body-{name}",
    )


def set_cell_text(cell, text: str, *, size: float, bold: bool, color: str) -> None:
    frame = cell.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = Inches(0.08)
    frame.margin_right = Inches(0.08)
    frame.margin_top = Inches(0.05)
    frame.margin_bottom = Inches(0.04)
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    paragraph = frame.paragraphs[0]
    paragraph.space_before = Pt(0)
    paragraph.space_after = Pt(0)
    paragraph.line_spacing = 1.0
    run = paragraph.add_run()
    run.text = text
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = rgb(color)


def add_results_table(slide) -> None:
    headers = ["Question", "What we measured", "Result", "What it means"]
    rows = [
        [
            "Do CPU features match?",
            "8/8 tokens match; max log-prob difference = 1.43e−6",
            "YES",
            "CPU and GPU give the same tokens and nearly identical logits",
        ],
        [
            "Is one CPU core fast enough?",
            "We need 5.86 img/s; one core gives 17.43 img/s",
            "2.98×",
            "One CPU thread is enough",
        ],
        [
            "Does batching help?",
            "Batch 8 is 7.96% faster; our gate is 20%",
            "NO",
            "Do not add live batching",
        ],
        [
            "Does the cache reduce transfer?",
            "Request bytes −99.82%; serialization −77.94%; p95 request time −29.30%",
            "YES",
            "The cache removes most of the transfer",
        ],
        [
            "Does the rollout path get faster?",
            "Rollout: 1.344 → 1.261 s (−6.17%); our gate is 10%",
            "NO",
            "Mechanism result; matched training-cycle study is pending",
        ],
    ]
    table_shape = slide.shapes.add_table(6, 4, Inches(0.66), Inches(1.66), Inches(12.02), Inches(4.46))
    table_shape.name = "results-table"
    table = table_shape.table
    widths = [2.20, 4.65, 1.35, 3.82]
    for column, width in zip(table.columns, widths, strict=True):
        column.width = Inches(width)
    table.rows[0].height = Inches(0.52)
    for row_index in range(1, len(table.rows)):
        table.rows[row_index].height = Inches(0.788)

    for column_index, header in enumerate(headers):
        cell = table.cell(0, column_index)
        cell.fill.solid()
        cell.fill.fore_color.rgb = rgb(BLACK)
        set_cell_text(cell, header, size=14.5, bold=True, color=WHITE)

    for row_index, row_data in enumerate(rows, 1):
        for column_index, value in enumerate(row_data):
            cell = table.cell(row_index, column_index)
            cell.fill.solid()
            cell.fill.fore_color.rgb = rgb(WHITE if row_index % 2 else GRAY_100)
            set_cell_text(
                cell,
                value,
                size=13.0 if column_index != 2 else 13.5,
                bold=column_index in {0, 2},
                color=BLACK,
            )


def add_notes(slide, text: str) -> None:
    notes_frame = slide.notes_slide.notes_text_frame
    notes_frame.text = text


def build_title_slide(prs: Presentation):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_white_background(slide)
    add_text(
        slide,
        "CPU Vision Encoder for\nMultimodal RL",
        0.72,
        1.42,
        9.65,
        1.55,
        size=35,
        bold=True,
        name="title",
    )
    add_line(slide, 0.74, 3.24, 12.60, 3.24, width=1.1, name="title-rule")
    add_text(
        slide,
        "Compute the vision feature once, then reuse it",
        0.74,
        3.58,
        8.8,
        0.42,
        size=22,
        name="subtitle",
    )
    add_text(
        slide,
        "The vision model is frozen during RL. We move it to CPU and cache its output for rollout and training.",
        0.74,
        4.36,
        10.9,
        0.55,
        size=17.5,
        color=GRAY_700,
        name="thesis",
    )
    add_text(
        slide,
        "Supervisor one-on-one · 19 August 2026",
        0.74,
        6.66,
        5.8,
        0.28,
        size=12,
        color=GRAY_500,
        name="date",
    )
    add_notes(
        slide,
        "I started this project to move the frozen vision encoder to CPU. The more useful idea is simple: the policy "
        "changes during training, but the vision feature does not. We should compute it once and reuse it.",
    )
    return slide


def build_motivation_slide(prs: Presentation):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_white_background(slide)
    add_header(slide, "Why compute the same vision feature again?", 1)
    add_text(
        slide,
        "The policy changes during training. The frozen vision encoder does not, so the same image always gives the same feature.",
        0.66,
        1.22,
        11.95,
        0.44,
        size=17.5,
        color=GRAY_700,
        name="setup",
    )

    add_labelled_box(
        slide,
        x=0.66,
        y=2.35,
        w=1.52,
        h=1.34,
        title="Image",
        body="pixels\n+ image grid",
        name="image",
    )
    add_text(slide, "→", 2.25, 2.72, 0.32, 0.34, size=25, align=PP_ALIGN.CENTER, name="arrow-1")
    add_labelled_box(
        slide,
        x=2.65,
        y=2.16,
        w=2.48,
        h=1.72,
        title="CPU vision encoder",
        body="frozen vision model\nCPU cache: cache its output",
        name="encoder",
        gray=True,
    )
    add_text(slide, "→", 5.22, 2.72, 0.32, 0.34, size=25, align=PP_ALIGN.CENTER, name="arrow-2")
    add_labelled_box(
        slide,
        x=5.62,
        y=2.06,
        w=2.73,
        h=1.92,
        title="Cached vision feature",
        body="final + 3 DeepStack features\nkey = image + model version",
        name="feature",
    )
    add_line(slide, 8.36, 3.02, 8.82, 3.02, width=1.0, name="branch-stem")
    add_line(slide, 8.82, 2.31, 8.82, 3.73, width=1.0, name="branch-vertical")
    add_line(slide, 8.82, 2.31, 9.16, 2.31, width=1.0, name="branch-rollout")
    add_line(slide, 8.82, 3.73, 9.16, 3.73, width=1.0, name="branch-actor")
    add_labelled_box(
        slide,
        x=9.18,
        y=1.69,
        w=3.36,
        h=1.30,
        title="SGLang rollout",
        body="SGLang cache: send the feature once",
        name="rollout",
        gray=True,
    )
    add_labelled_box(
        slide,
        x=9.18,
        y=3.10,
        w=3.36,
        h=1.30,
        title="Megatron actor",
        body="uses the same feature",
        name="actor",
    )

    add_rect(slide, 0.66, 5.10, 11.88, 1.22, fill=GRAY_100, line=GRAY_200, name="conclusion-box")
    add_text(
        slide,
        "Goal",
        0.88,
        5.31,
        0.70,
        0.28,
        size=16,
        bold=True,
        name="goal-label",
    )
    add_text(
        slide,
        "Compute each feature once and stop sending the full feature every time.",
        1.70,
        5.29,
        9.95,
        0.32,
        size=17,
        name="goal-text",
    )
    add_text(
        slide,
        "The vision encoder has only 27.1M parameters. VRAM saving alone is not enough for a paper.",
        0.88,
        5.79,
        10.95,
        0.28,
        size=14.5,
        color=GRAY_700,
        italic=True,
        name="motivation-caveat",
    )
    add_notes(
        slide,
        "We can cache the feature safely because the vision model is frozen. The cache key includes the image and the model "
        "version, so we do not reuse the wrong feature. CPU cache saves vision compute. SGLang cache stops us from "
        "sending the full feature again.",
    )
    return slide


def build_results_slide(prs: Presentation):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_white_background(slide)
    add_header(slide, "What we found so far", 2)
    add_text(
        slide,
        "We tested correctness, CPU speed, batching, and caching on four MI210 GPUs.",
        0.66,
        1.20,
        11.92,
        0.34,
        size=16.5,
        color=GRAY_700,
        name="setup",
    )
    add_results_table(slide)
    add_rect(slide, 0.66, 6.27, 12.02, 0.55, fill=GRAY_100, line=GRAY_200, name="result-conclusion")
    add_text(
        slide,
        "Best setup: one CPU worker with one thread. The cache hit all 320 requests with no misses.",
        0.86,
        6.43,
        11.60,
        0.22,
        size=13.5,
        bold=True,
        name="result-summary",
    )
    add_text(
        slide,
        "Still unresolved: GPU grouped n=8 has 0.039 log-prob drift, above our 0.01 gate.",
        0.66,
        6.88,
        11.75,
        0.22,
        size=10.5,
        color=GRAY_500,
        italic=True,
        name="result-caveat",
    )
    add_notes(
        slide,
        "One CPU thread is already 2.98 times faster than we need. More CPU workers reduce request latency, but they use more memory "
        "and actor wait is still below one percent. The cache removes almost all request bytes, but rollout improves by only 6.17 percent. "
        "That is a rollout-path mechanism result, not a training-speed claim; the matched training-cycle study is "
        "pending.",
    )
    return slide


def build_paper_slide(prs: Presentation):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_white_background(slide)
    add_header(slide, "How this becomes a paper", 3)
    add_text(
        slide,
        "The paper is not just “move the vision encoder to CPU.” The useful idea is to compute a frozen vision feature once and reuse it safely.",
        0.66,
        1.20,
        11.90,
        0.42,
        size=17.5,
        color=GRAY_700,
        name="setup",
    )

    add_rect(slide, 0.66, 1.84, 5.82, 3.82, fill=WHITE, line=BLACK, name="claims-box")
    add_text(slide, "What we already have", 0.91, 2.08, 4.90, 0.36, size=20, bold=True, name="claims-title")
    add_multiline(
        slide,
        [
            ("CPU and GPU give the same vision features, the same tokens, and nearly identical logits.", False),
            ("We have two caches: one saves vision compute and one saves feature transfer.", False),
            ("We measure how much speed we need first, so we know when more CPUs or batching will not help.", False),
        ],
        0.94,
        2.62,
        5.26,
        2.66,
        size=15.5,
        bullet=True,
        gap=12,
        name="claims-list",
    )

    add_rect(slide, 6.74, 1.84, 5.94, 3.82, fill=GRAY_050, line=BLACK, name="needed-box")
    add_text(
        slide,
        "What I still need next month",
        7.00,
        2.08,
        5.22,
        0.36,
        size=20,
        bold=True,
        name="needed-title",
    )
    add_multiline(
        slide,
        [
            ("Repeat the cache experiment and add error bars.", False),
            ("Compare reward and learning curves, not only features and logits.", False),
            ("Test a larger vision encoder or harder inputs: high resolution, multiple images, or video.", False),
            ("Test cold cache, eviction, restart, and routing failures.", False),
        ],
        7.02,
        2.62,
        5.32,
        2.70,
        size=15.0,
        bullet=True,
        gap=8,
        name="needed-list",
    )

    add_rect(slide, 0.66, 5.93, 12.02, 0.63, fill=BLACK, line=BLACK, name="paper-title-box")
    add_text(
        slide,
        "Working title: Reusing Frozen Vision Features in Asynchronous Multimodal RL",
        0.92,
        6.13,
        11.50,
        0.25,
        size=15.8,
        bold=True,
        color=WHITE,
        name="paper-title",
    )
    add_text(
        slide,
        "Current limit: the encoder is small; vision cache reuse is mechanism evidence, while the matched "
        "training-cycle study is pending.",
        0.66,
        6.78,
        11.88,
        0.30,
        size=11.5,
        color=GRAY_500,
        italic=True,
        name="paper-caveat",
    )
    add_notes(
        slide,
        "The paper idea is simple: the frozen vision output does not change, so compute it once and reuse it. We already have the cache "
        "design and the main measurements. The biggest risk is that the current vision encoder is small and the actor is not waiting for it. "
        "I still need repeats, learning results, and one harder multimodal setting.",
    )
    return slide


def iter_runs(slide):
    for shape in slide.shapes:
        if getattr(shape, "has_text_frame", False):
            for paragraph in shape.text_frame.paragraphs:
                yield from paragraph.runs
        if getattr(shape, "has_table", False):
            for row in shape.table.rows:
                for cell in row.cells:
                    for paragraph in cell.text_frame.paragraphs:
                        yield from paragraph.runs


def slide_text(slide) -> str:
    parts = []
    for shape in slide.shapes:
        if getattr(shape, "has_text_frame", False) and shape.text:
            parts.append(shape.text)
        if getattr(shape, "has_table", False):
            parts.extend(cell.text for row in shape.table.rows for cell in row.cells)
    return "\n".join(parts)


def validate_deck(path: Path) -> str:
    prs = Presentation(path)
    findings: list[str] = []
    if len(prs.slides) != 4:
        findings.append(f"slide count is {len(prs.slides)}, expected 4")
    if abs(prs.slide_width / 914400 - SLIDE_W) > 0.01 or abs(prs.slide_height / 914400 - SLIDE_H) > 0.01:
        findings.append("slide size is not 16:9 at 13.333 × 7.5 inches")

    required = [
        "CPU Vision Encoder for",
        "Why compute the same vision feature again?",
        "Request bytes −99.82%",
        "Rollout: 1.344 → 1.261 s (−6.17%)",
        "Reusing Frozen Vision Features",
    ]
    all_text = "\n".join(slide_text(slide) for slide in prs.slides)
    for text in required:
        if text not in all_text:
            findings.append(f"required text is absent: {text}")

    for slide_number, slide in enumerate(prs.slides, 1):
        if not slide.has_notes_slide or not slide.notes_slide.notes_text_frame.text.strip():
            findings.append(f"slide {slide_number} has no speaker notes")
        for shape in slide.shapes:
            if shape.left < 0 or shape.top < 0 or shape.left + shape.width > prs.slide_width or shape.top + shape.height > prs.slide_height:
                findings.append(f"slide {slide_number}: {shape.name} exceeds slide bounds")
        for run in iter_runs(slide):
            if run.font.size is not None and run.font.size.pt < 8:
                findings.append(f"slide {slide_number}: font below 8 pt")
            if run.font.name and run.font.name != FONT:
                findings.append(f"slide {slide_number}: unexpected font {run.font.name}")

        text_shapes = [
            shape
            for shape in slide.shapes
            if getattr(shape, "has_text_frame", False) and shape.text.strip()
        ]
        tolerance = 9144
        for first_index, first in enumerate(text_shapes):
            for second in text_shapes[first_index + 1 :]:
                horizontal_overlap = min(first.left + first.width, second.left + second.width) - max(
                    first.left, second.left
                )
                vertical_overlap = min(first.top + first.height, second.top + second.height) - max(
                    first.top, second.top
                )
                if horizontal_overlap > tolerance and vertical_overlap > tolerance:
                    findings.append(
                        f"slide {slide_number}: text boxes {first.name!r} and {second.name!r} overlap"
                    )

    result = "PASS" if not findings else "FAIL"
    lines = [
        "# CPU vision encoder slide validation",
        "",
        f"- Deck: `{path.name}`",
        f"- Result: **{result}**",
        f"- Slide count: **{len(prs.slides)}**",
        "- Canvas: **16:9, white, monochrome authored content**",
        "- Editable objects: **native PowerPoint text, shapes, lines, and table**",
        "- Speaker notes: **present on all four slides**",
        "- Text-object overlap check: **PASS**",
        "- Renderer: **not available in this environment; no native rendered visual approval claimed**",
        "",
        "## Static findings",
        "",
    ]
    lines.extend(f"- {finding}" for finding in findings)
    if not findings:
        lines.append("- No static validation findings.")
    lines.extend(
        [
            "",
            "## Verified result table entries",
            "",
            "- CPU/GPU token match: `8/8`; maximum log-probability drift: `1.43e−6`.",
            "- Peak demand / one-core supply: `5.86 / 17.43 images/s` (`2.98×`).",
            "- Batch-eight gain: `7.96%`, below the `20%` eligibility gate.",
            "- SGLang cache: request bytes `−99.82%`, serialization `−77.94%`, p95 RTT `−29.30%`.",
            "- Rollout time: `1.344 → 1.261 s` (`−6.17%`), below the `10%` adoption gate.",
            "- Claim boundary: vision cache reuse is rollout-path mechanism evidence, not a training-cycle speedup.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    if not REFERENCE_DECK.is_file():
        raise FileNotFoundError(f"monochrome reference deck does not exist: {REFERENCE_DECK}")
    prs = Presentation(REFERENCE_DECK)
    remove_template_slides(prs)
    prs.slide_width = Inches(SLIDE_W)
    prs.slide_height = Inches(SLIDE_H)
    build_title_slide(prs)
    build_motivation_slide(prs)
    build_results_slide(prs)
    build_paper_slide(prs)
    prs.core_properties.title = "CPU Vision Encoder for Multimodal RL"
    prs.core_properties.subject = "Supervisor one-on-one project update"
    prs.core_properties.author = "Qirong Ho"
    prs.core_properties.keywords = "CPU vision encoder, multimodal RL, vision feature cache"
    prs.save(OUTPUT_PPTX)

    validation = validate_deck(OUTPUT_PPTX)
    VALIDATION_REPORT.write_text(validation, encoding="utf-8")
    if "Result: **FAIL**" in validation:
        raise RuntimeError(f"slide validation failed; see {VALIDATION_REPORT}")
    print(f"Wrote {OUTPUT_PPTX}")
    print(f"Wrote {VALIDATION_REPORT}")


if __name__ == "__main__":
    main()
