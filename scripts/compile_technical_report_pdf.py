"""
Publication-Grade Technical Report PDF Compiler
Candidate: Faizan Iqbal
Target Role: AI Research Engineer, Bodhan AI / AI4Bharat (IIT Madras)
Strict Academic Formatting: Zero Em Dashes, Clean Captions, Formal Layout
"""

import sys
import os
import re
import html

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable, Preformatted
)
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ---------------------------------------------------------
# FONT REGISTRATION
# ---------------------------------------------------------
fonts_dir = os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts')
nirmala_path = os.path.join(fonts_dir, 'Nirmala.ttc')
consola_path = os.path.join(fonts_dir, 'consola.ttf')

if os.path.exists(nirmala_path):
    pdfmetrics.registerFont(TTFont('Nirmala', nirmala_path, subfontIndex=3))       # Regular
    pdfmetrics.registerFont(TTFont('Nirmala-Bold', nirmala_path, subfontIndex=4))  # Bold
    pdfmetrics.registerFontFamily('Nirmala', normal='Nirmala', bold='Nirmala-Bold', italic='Nirmala', boldItalic='Nirmala-Bold')
    BODY_FONT = 'Nirmala'
    BOLD_FONT = 'Nirmala-Bold'
else:
    BODY_FONT = 'Helvetica'
    BOLD_FONT = 'Helvetica-Bold'

if os.path.exists(consola_path):
    pdfmetrics.registerFont(TTFont('Consolas', consola_path))
    CODE_FONT = 'Consolas'
else:
    CODE_FONT = 'Courier'

print(f"Fonts loaded: Body={BODY_FONT}, Code={CODE_FONT}")

# ---------------------------------------------------------
# NUMBERED CANVAS (Running Header and Footer)
# ---------------------------------------------------------
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        page_w, page_h = self._pagesize

        # Running Header on page 2+
        if self._pageNumber > 1:
            self.setFont(BODY_FONT, 7.5)
            self.setFillColor(colors.HexColor('#4A5568'))
            self.drawString(36, page_h - 22, "AI Research Engineer Take-Home Assignment | Bodhan AI / AI4Bharat (IIT Madras)")
            self.drawRightString(page_w - 36, page_h - 22, "Technical Report: Fine-Tuning Indic-Canary for Marathi ASR")
            self.setStrokeColor(colors.HexColor('#CBD5E0'))
            self.setLineWidth(0.5)
            self.line(36, page_h - 25, page_w - 36, page_h - 25)

        # Running Footer on all pages
        self.setFont(BODY_FONT, 7.5)
        self.setFillColor(colors.HexColor('#718096'))
        self.drawString(36, 18, "Faizan Iqbal: Parameter-Efficient Fine-Tuning of Indic-Canary for Marathi ASR")
        self.drawRightString(page_w - 36, 18, f"Page {self._pageNumber} of {page_count}")
        self.setStrokeColor(colors.HexColor('#E2E8F0'))
        self.setLineWidth(0.5)
        self.line(36, 26, page_w - 36, 26)

        self.restoreState()


# ---------------------------------------------------------
# STYLES SETUP
# ---------------------------------------------------------
PRIMARY_COLOR = colors.HexColor('#1A365D')    # Deep Navy
SECONDARY_COLOR = colors.HexColor('#2B6CB0')  # Slate Blue
TEXT_COLOR = colors.HexColor('#2D3748')       # Dark Charcoal
BG_LIGHT = colors.HexColor('#F7FAFC')         # Off-white / light slate
BORDER_COLOR = colors.HexColor('#CBD5E0')

title_style = ParagraphStyle(
    'DocTitle',
    fontName=BOLD_FONT,
    fontSize=16.0,
    leading=20.0,
    textColor=PRIMARY_COLOR,
    spaceAfter=4,
)

subtitle_style = ParagraphStyle(
    'DocSubtitle',
    fontName=BODY_FONT,
    fontSize=9.5,
    leading=13.0,
    textColor=SECONDARY_COLOR,
    spaceAfter=6,
)

meta_style = ParagraphStyle(
    'MetaText',
    fontName=BODY_FONT,
    fontSize=8.0,
    leading=11.5,
    textColor=TEXT_COLOR,
)

h1_style = ParagraphStyle(
    'H1_Custom',
    fontName=BOLD_FONT,
    fontSize=11.5,
    leading=14.5,
    textColor=PRIMARY_COLOR,
    spaceBefore=10,
    spaceAfter=3,
    keepWithNext=True,
)

h2_style = ParagraphStyle(
    'H2_Custom',
    fontName=BOLD_FONT,
    fontSize=9.8,
    leading=12.8,
    textColor=SECONDARY_COLOR,
    spaceBefore=8,
    spaceAfter=2.5,
    keepWithNext=True,
)

h3_style = ParagraphStyle(
    'H3_Custom',
    fontName=BOLD_FONT,
    fontSize=8.8,
    leading=11.8,
    textColor=TEXT_COLOR,
    spaceBefore=6,
    spaceAfter=2,
    keepWithNext=True,
)

body_style = ParagraphStyle(
    'Body_Custom',
    fontName=BODY_FONT,
    fontSize=8.4,
    leading=11.6,
    textColor=TEXT_COLOR,
    spaceAfter=3.0,
)

bullet_style = ParagraphStyle(
    'Bullet_Custom',
    fontName=BODY_FONT,
    fontSize=8.4,
    leading=11.6,
    textColor=TEXT_COLOR,
    leftIndent=14,
    firstLineIndent=-9,
    spaceAfter=2.5,
)

abstract_style = ParagraphStyle(
    'Abstract_Custom',
    fontName=BODY_FONT,
    fontSize=8.2,
    leading=11.5,
    textColor=TEXT_COLOR,
    alignment=4, # Justified
)

caption_style = ParagraphStyle(
    'Caption_Custom',
    fontName=BOLD_FONT,
    fontSize=7.8,
    leading=10.2,
    textColor=SECONDARY_COLOR,
    spaceBefore=5,
    spaceAfter=2,
    keepWithNext=True,
)

table_cell_style = ParagraphStyle(
    'TableCell',
    fontName=BODY_FONT,
    fontSize=7.6,
    leading=9.8,
    textColor=TEXT_COLOR,
)

table_header_style = ParagraphStyle(
    'TableHeader',
    fontName=BOLD_FONT,
    fontSize=7.8,
    leading=10.2,
    textColor=colors.white,
)

code_style = ParagraphStyle(
    'CodeBlock',
    fontName=CODE_FONT,
    fontSize=6.8,
    leading=8.6,
    textColor=colors.HexColor('#1A202C'),
)

quote_style = ParagraphStyle(
    'QuoteText',
    fontName=BODY_FONT,
    fontSize=8.4,
    leading=11.6,
    textColor=colors.HexColor('#2C5282'),
)

math_box_style = ParagraphStyle(
    'MathBox',
    fontName=BODY_FONT,
    fontSize=8.2,
    leading=10.8,
    textColor=colors.HexColor('#1A365D'),
    alignment=1, # Centered
)


def has_indic_chars(text):
    return any('\u0900' <= ch <= '\u097f' for ch in text)


def clean_markdown_inline(text):
    """Safely converts markdown inline formatting to reportlab tags with font-aware Indic support and ZERO em dashes."""
    if not text:
        return ""
    
    # Strip any em dash or en dash to ensure strict adherence
    text = text.replace('\u2014', ', ').replace('\u2013', '-')

    # Strip TeX-style backslash before percent
    text = text.replace(r'\%', '%')

    # Convert TeX exponents: 10^{-4} -> 10^-4
    text = re.sub(r'10\^\{([^\}]+)\}', r'10<sup>\1</sup>', text)

    # Convert known math notations safely
    text = text.replace(r'\mathcal{L}_{\text{CE}}', 'L<sub>CE</sub>')
    text = text.replace(r'\mathcal{L}', 'L')
    text = text.replace(r'\mathbf{T}_{\text{prompt}}', 'T_prompt')
    text = text.replace(r'\mathbf{Y}_{\text{full}}', 'Y_full')
    text = text.replace(r'\mathbf{Y}_{\text{dec\_in}}', 'Y_dec_in')
    text = text.replace(r'\mathbf{Y}_{\text{labels}}', 'Y_labels')
    text = text.replace(r'\mathbf{X}_{\text{audio}}', 'X_audio')
    text = text.replace(r'\eta_{\max}', 'η_max')
    text = text.replace(r'\eta_{\min}', 'η_min')
    text = text.replace(r'\beta_1', 'β1')
    text = text.replace(r'\beta_2', 'β2')
    text = text.replace(r'\epsilon', 'ε')
    text = text.replace(r'\alpha', 'α')
    text = text.replace(r'\pm', '±')
    text = text.replace(r'\times', '×')
    text = text.replace(r'\implies', '⇒')
    text = text.replace(r'\to', '→')
    text = text.replace(r'\le', '≤')
    text = text.replace(r'\ge', '≥')
    text = text.replace(r'\text{Train}', 'Train')
    text = text.replace(r'\text{Val}', 'Val')
    text = text.replace(r'\mathcal{S}_{\text{train\_exclusive}}', 'S_train_exclusive')
    text = text.replace(r'\mathcal{S}_{\text{official\_train}}', 'S_official_train')
    text = text.replace(r'\mathcal{S}_{\text{official\_val}}', 'S_official_val')
    text = text.replace(r'\mathcal{S}_{\text{val\_benchmark}}', 'S_val_benchmark')
    text = text.replace(r'\mathcal{S}_{\text{text}}', 'S_text')
    text = text.replace(r'd_{\text{model}}', 'd_model')
    text = text.replace(r'\dots', '...')
    text = text.replace(r'\setminus', '\\')
    
    # Specific clean substitutions for math sets
    text = text.replace(r'\cap', '∩')
    text = text.replace(r'\emptyset', '∅')

    # Extract links first to avoid escaping <a href>
    links = []
    def link_repl(match):
        idx = len(links)
        links.append((match.group(1), match.group(2)))
        return f"__LINK_TOKEN_{idx}__"
    text = re.sub(r'\[([^\]]+)\]\(([^\)]+)\)', link_repl, text)

    # Extract inline code backticks `code`
    codes = []
    def code_repl(match):
        idx = len(codes)
        code_str = match.group(1)
        codes.append(code_str)
        return f"__CODE_TOKEN_{idx}__"
    text = re.sub(r'`([^`]+)`', code_repl, text)

    # Clean leftover LaTeX math dollar signs $...$
    maths = []
    def math_repl(match):
        idx = len(maths)
        maths.append(match.group(1))
        return f"__MATH_TOKEN_{idx}__"
    text = re.sub(r'\$([^\$]+)\$', math_repl, text)

    # HTML Escape raw text
    text = html.escape(text)

    # Bold **text**
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    # Italic *text*
    text = re.sub(r'(?<!\*)\*(?!\*)(.*?)(?<!\*)\*(?!\*)', r'<i>\1</i>', text)

    # Restore math
    for idx, m in enumerate(maths):
        m_clean = m.replace('{', '').replace('}', '')
        escaped_m = html.escape(m_clean)
        text = text.replace(f"__MATH_TOKEN_{idx}__", f"<i>{escaped_m}</i>")

    # Restore inline code (Font-aware: Nirmala bold for Indic, Consolas for ASCII/code)
    for idx, c in enumerate(codes):
        escaped_c = html.escape(c)
        if has_indic_chars(c):
            text = text.replace(f"__CODE_TOKEN_{idx}__", f'<b><font color="#4C1D95">{escaped_c}</font></b>')
        else:
            text = text.replace(f"__CODE_TOKEN_{idx}__", f'<font name="{CODE_FONT}" color="#6B46C1">{escaped_c}</font>')

    # Restore links
    for idx, (ltitle, lurl) in enumerate(links):
        escaped_title = html.escape(ltitle)
        text = text.replace(f"__LINK_TOKEN_{idx}__", f'<a href="{lurl}"><u><font color="#2B6CB0">{escaped_title}</font></u></a>')

    return text


def clean_math_display(line_text):
    """Converts display LaTeX equations into clear, elegant typography strings with priority order."""
    t = line_text.strip()
    if t.startswith('$$') and t.endswith('$$'):
        t = t[2:-2].strip()

    # Labels / Masking
    if r'\mathbf{Y}_{\text{labels}}' in t or r'\begin{cases}' in t:
        return "<b>Loss Masking:</b> &nbsp; <b>Y</b><sub>labels</sub>[t] = <b>-100</b> &nbsp; if t &lt; |<b>T</b><sub>prompt</sub>| - 1, &nbsp; else &nbsp; <b>Y</b><sub>full</sub>[t+1]"
    
    # Decoder Input
    if r'\mathbf{Y}_{\text{dec\_in}}' in t:
        return "<b>Decoder Input:</b> &nbsp; <b>Y</b><sub>dec_in</sub> = <b>Y</b><sub>full</sub>[:-1] &nbsp; <i>(Right-shifted teacher-forcing sequence)</i>"

    # Target Sequence
    if r'\mathbf{Y}_{\text{full}}' in t:
        return "<b>Target Sequence:</b> &nbsp; <b>Y</b><sub>full</sub> = [ <b>T</b><sub>prompt</sub>, y<sub>1</sub>, y<sub>2</sub>, ..., y<sub>M</sub>, &lt;eos&gt; ]"

    # Prompt Vector
    if r'\mathbf{T}_{\text{prompt}}' in t:
        return "<b>Prompt Vector:</b> &nbsp; <b>T</b><sub>prompt</sub> = [ &lt;startoftranscript&gt;, &lt;lang=mr&gt;, &lt;transcribe&gt;, &lt;no_pnc&gt; ]"

    # Cross-Entropy Loss
    if r'\mathcal{L}' in t or r'\sum' in t:
        return "<b>Cross-Entropy Loss:</b> &nbsp; <i>L</i><sub>CE</sub> = - (1 / |S<sub>text</sub>|) · Σ<sub>t ∈ S<sub>text</sub></sub> [ (1 - ε) · log P(y<sub>t</sub> | <b>Y</b><sub>&lt;t</sub>, <b>X</b><sub>audio</sub>) + (ε / V) · Σ<sub>c=1</sub><sup>V</sup> log P(c | <b>Y</b><sub>&lt;t</sub>, <b>X</b><sub>audio</sub>) ]"

    # Low-Rank Decomposition
    if r'\Delta W' in t or r'\frac{\alpha}{r}' in t or r'W_0' in t:
        return "<b>Low-Rank Decomposition:</b> &nbsp; <b>W</b> = <b>W</b><sub>0</sub> + Δ<b>W</b> = <b>W</b><sub>0</sub> + (α / r) · <b>B</b> · <b>A</b> &nbsp; <i>(with r=16, α=32, multiplier=2.0)</i>"

    # Speaker Disjoint Set
    if r'\mathcal{S}_{\text{train\_exclusive}}' in t or r'\setminus' in t:
        return "<b>Speaker Partitioning:</b> &nbsp; S<sub>train_exclusive</sub> = S<sub>official_train</sub> \\ S<sub>official_val</sub> &nbsp; (103 Exclusive Speakers)"

    if r'\mathcal{S}_{\text{val\_benchmark}}' in t or r'\cap' in t:
        return f"<b>Speaker Leakage Guard:</b> &nbsp; <font name='{CODE_FONT}'>len(train_speakers &amp; val_speakers) == 0</font> &nbsp; <i>(Strictly Disjoint: 0 Overlap)</i>"

    # Generic fallback
    return clean_markdown_inline(t)


def build_table_from_markdown(md_table_text, printable_width=523):
    lines = [l.strip() for l in md_table_text.strip().split('\n') if l.strip()]
    if len(lines) < 2:
        return None
    
    header_raw = [c.strip() for c in lines[0].split('|')[1:-1]]
    data_rows_raw = []
    for line in lines[2:]:
        cols = [c.strip() for c in line.split('|')[1:-1]]
        if cols:
            data_rows_raw.append(cols)
            
    num_cols = len(header_raw)
    if num_cols == 0:
        return None
    
    # Calculate column widths
    if num_cols == 7:
        if 'Dataset Partition' in header_raw[0]:
            col_widths = [115, 62, 68, 65, 75, 53, 85]
        else:
            col_widths = [65, 50, 55, 65, 65, 65, 158]
    elif num_cols == 6:
        if 'Scheduled Learning Rate' in header_raw[1]:
            col_widths = [82, 85, 92, 85, 93, 86]
        else:
            col_widths = [75, 55, 62, 105, 115, 111]
    elif num_cols == 4:
        if 'Obstacle' in header_raw[0]:
            col_widths = [105, 115, 140, 163]
        elif 'Metric' in header_raw[0]:
            col_widths = [173, 115, 125, 110]
        else:
            col_widths = [148, 125, 125, 125]
    elif num_cols == 3:
        if 'Demographic' in header_raw[0]:
            col_widths = [173, 175, 175]
        else:
            col_widths = [143, 190, 190]
    else:
        col_widths = [printable_width / num_cols] * num_cols

    total_w = sum(col_widths)
    col_widths = [w * (printable_width / total_w) for w in col_widths]

    table_data = []
    h_row = []
    for c in header_raw:
        clean_text = clean_markdown_inline(c)
        h_row.append(Paragraph(clean_text, table_header_style))
    table_data.append(h_row)

    for r in data_rows_raw:
        row_cells = []
        for c_idx in range(num_cols):
            text = r[c_idx] if c_idx < len(r) else ''
            clean_text = clean_markdown_inline(text)
            row_cells.append(Paragraph(clean_text, table_cell_style))
        table_data.append(row_cells)

    t = Table(table_data, colWidths=col_widths, repeatRows=1)
    
    t_style = [
        ('BACKGROUND', (0, 0), (-1, 0), SECONDARY_COLOR),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 2.2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.2),
        ('LEFTPADDING', (0, 0), (-1, -1), 3.5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 3.5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
    ]

    for i in range(1, len(table_data)):
        if i % 2 == 0:
            t_style.append(('BACKGROUND', (0, i), (-1, i), BG_LIGHT))
        else:
            t_style.append(('BACKGROUND', (0, i), (-1, i), colors.white))

    t.setStyle(TableStyle(t_style))
    return t


def compile_pdf(source_md_path, target_pdf_path):
    print(f"Reading markdown source: {source_md_path}")
    with open(source_md_path, 'r', encoding='utf-8') as f:
        md_text = f.read()

    doc = SimpleDocTemplate(
        target_pdf_path,
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=30,
        bottomMargin=32,
    )
    printable_w = A4[0] - 72

    story = []

    # ---------------------------------------------------------
    # PAGE 1: FORMAL ACADEMIC TITLE & METADATA
    # ---------------------------------------------------------
    title_p = Paragraph(
        "Technical Report: Parameter-Efficient Fine-Tuning of Indic-Canary for Marathi Automatic Speech Recognition",
        title_style
    )
    story.append(title_p)
    
    subtitle_p = Paragraph(
        "Architectural Investigation, Corpus Leakage Mitigation, and Parameter-Efficient Adaptation Under Hardware Constraints",
        subtitle_style
    )
    story.append(subtitle_p)
    story.append(HRFlowable(width="100%", thickness=1.5, color=PRIMARY_COLOR, spaceBefore=1, spaceAfter=6))

    # Metadata Table
    meta_data = [
        [
            Paragraph("<b>Candidate:</b> Faizan Iqbal<br/>"
                      "<b>Target Role:</b> AI Research Engineer, Bodhan AI / AI4Bharat (IIT Madras)<br/>"
                      "<b>Faculty Lead / Evaluation:</b> Prof. Mitesh M. Khapra", meta_style),
            Paragraph("<b>Date:</b> September 2026<br/>"
                      "<b>Repository:</b> <a href='https://github.com/Faizaniqbal52/indic-canary-marathi-asr'><u>github.com/Faizaniqbal52/indic-canary-marathi-asr</u></a><br/>"
                      "<b>Artifacts (Drive):</b> <a href='https://drive.google.com/drive/folders/1STc13nhFeo2_pD9FL3clTzBXPMskG_a5?usp=drive_link'><u>Google Drive Submission Package</u></a>", meta_style)
        ]
    ]
    meta_table = Table(meta_data, colWidths=[270, 253])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), BG_LIGHT),
        ('BOX', (0, 0), (-1, -1), 0.75, colors.HexColor('#CBD5E0')),
        ('TOPPADDING', (0, 0), (-1, -1), 4.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 6.5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6.5),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 8))

    lines = md_text.split('\n')
    
    in_abstract = False
    abstract_paras = []
    
    in_code_block = False
    code_block_lines = []

    in_table = False
    table_lines = []

    start_parsing = False

    i = 0
    while i < len(lines):
        line = lines[i]

        if line.startswith('## Abstract'):
            start_parsing = True
            in_abstract = True
            i += 1
            continue

        if not start_parsing:
            i += 1
            continue

        # Code block fence
        if line.strip().startswith('```'):
            if not in_code_block:
                in_code_block = True
                code_block_lines = []
            else:
                in_code_block = False
                code_text = '\n'.join(code_block_lines)
                code_p = Preformatted(code_text, code_style)
                code_box = Table([[code_p]], colWidths=[printable_w])
                code_box.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8FAFC')),
                    ('BOX', (0, 0), (-1, -1), 0.75, colors.HexColor('#CBD5E0')),
                    ('TOPPADDING', (0, 0), (-1, -1), 3.5),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
                    ('LEFTPADDING', (0, 0), (-1, -1), 5),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 5),
                ]))
                story.append(Spacer(1, 1.5))
                story.append(code_box)
                story.append(Spacer(1, 3))
            i += 1
            continue

        if in_code_block:
            code_block_lines.append(line)
            i += 1
            continue

        # Markdown Table detection
        if line.strip().startswith('|') and '|' in line.strip()[1:]:
            if not in_table:
                in_table = True
                table_lines = [line]
            else:
                table_lines.append(line)
            i += 1
            continue
        else:
            if in_table:
                in_table = False
                t_obj = build_table_from_markdown('\n'.join(table_lines), printable_w)
                if t_obj:
                    story.append(Spacer(1, 2))
                    story.append(t_obj)
                    story.append(Spacer(1, 3.5))
                table_lines = []

        # Display math $$...$$
        if line.strip().startswith('$$'):
            math_content = line.strip()
            if not math_content.endswith('$$') or len(math_content) == 2:
                while i + 1 < len(lines) and not lines[i+1].strip().endswith('$$'):
                    i += 1
                    math_content += ' ' + lines[i].strip()
                if i + 1 < len(lines):
                    i += 1
                    math_content += ' ' + lines[i].strip()
            
            clean_m = clean_math_display(math_content)
            m_p = Paragraph(clean_m, math_box_style)
            m_box = Table([[m_p]], colWidths=[printable_w])
            m_box.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#EDF2F7')),
                ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E0')),
                ('TOPPADDING', (0, 0), (-1, -1), 3),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ]))
            story.append(Spacer(1, 1.5))
            story.append(m_box)
            story.append(Spacer(1, 3))
            i += 1
            continue

        # Abstract closure when hitting Section 1
        if in_abstract and line.startswith('## 1.'):
            in_abstract = False
            abs_flowables = [
                Paragraph("<b>ABSTRACT</b>", ParagraphStyle('AbsTitle', fontName=BOLD_FONT, fontSize=9.0, textColor=PRIMARY_COLOR, spaceAfter=2.5)),
                HRFlowable(width="100%", thickness=1, color=PRIMARY_COLOR, spaceBefore=1, spaceAfter=3.5),
            ]
            for ap in abstract_paras:
                abs_flowables.append(Paragraph(ap, abstract_style))
                abs_flowables.append(Spacer(1, 2.5))
            
            abs_table = Table([[abs_flowables]], colWidths=[printable_w])
            abs_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F0F4F8')),
                ('BOX', (0, 0), (-1, -1), 1, PRIMARY_COLOR),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('LEFTPADDING', (0, 0), (-1, -1), 7),
                ('RIGHTPADDING', (0, 0), (-1, -1), 7),
            ]))
            story.append(abs_table)
            story.append(Spacer(1, 10))
            # Formal academic title page break
            story.append(PageBreak())

        if in_abstract:
            stripped = line.strip()
            if stripped and not stripped.startswith('---'):
                cleaned = clean_markdown_inline(stripped)
                abstract_paras.append(cleaned)
            i += 1
            continue

        # Table or Figure Caption (lines like *Table 1: ...* or *Figure 1: ...*)
        if (line.strip().startswith('*Table ') and line.strip().endswith('*')) or (line.strip().startswith('*Figure ') and line.strip().endswith('*')):
            cap_text = line.strip().strip('*').strip()
            clean_cap = clean_markdown_inline(cap_text)
            story.append(Spacer(1, 2))
            story.append(Paragraph(f"<b>{clean_cap}</b>", caption_style))
            i += 1
            continue

        # Horizontal rule
        if line.strip() == '---':
            story.append(Spacer(1, 1.5))
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#CBD5E0'), spaceBefore=1, spaceAfter=1.5))
            i += 1
            continue

        # Headings
        if line.startswith('## '):
            h_text = clean_markdown_inline(line[3:].strip())
            story.append(Spacer(1, 3))
            story.append(Paragraph(h_text, h1_style))
            story.append(HRFlowable(width="100%", thickness=0.75, color=SECONDARY_COLOR, spaceBefore=1, spaceAfter=2.5))
            i += 1
            continue

        if line.startswith('### '):
            h_text = clean_markdown_inline(line[4:].strip())
            story.append(Paragraph(h_text, h2_style))
            i += 1
            continue

        if line.startswith('#### '):
            h_text = clean_markdown_inline(line[5:].strip())
            story.append(Paragraph(h_text, h3_style))
            i += 1
            continue

        # Blockquote >
        if line.strip().startswith('>'):
            raw_quote = line.strip()[1:].strip().strip(' *"')
            quote_text = clean_markdown_inline(raw_quote)
            q_p = Paragraph(f"<i>“{quote_text}”</i>", quote_style)
            q_table = Table([[q_p]], colWidths=[printable_w])
            q_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#EDF2F7')),
                ('LINEBEFORE', (0, 0), (0, -1), 3, PRIMARY_COLOR),
                ('TOPPADDING', (0, 0), (-1, -1), 3.5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                ('RIGHTPADDING', (0, 0), (-1, -1), 5),
            ]))
            story.append(Spacer(1, 1.5))
            story.append(q_table)
            story.append(Spacer(1, 2.5))
            i += 1
            continue

        # Bullet list
        if line.strip().startswith('* ') or line.strip().startswith('- '):
            bullet_text = clean_markdown_inline(line.strip()[2:].strip())
            story.append(Paragraph(f"• &nbsp; {bullet_text}", bullet_style))
            i += 1
            continue

        # Numbered list
        num_match = re.match(r'^\s*(\d+)\.\s+(.*)$', line)
        if num_match:
            idx = num_match.group(1)
            item_text = clean_markdown_inline(num_match.group(2))
            story.append(Paragraph(f"<b>{idx}.</b> &nbsp; {item_text}", bullet_style))
            i += 1
            continue

        # Regular Body Paragraph
        stripped = line.strip()
        if stripped:
            cleaned = clean_markdown_inline(stripped)
            story.append(Paragraph(cleaned, body_style))
        else:
            story.append(Spacer(1, 1.2))

        i += 1

    if in_table and table_lines:
        t_obj = build_table_from_markdown('\n'.join(table_lines), printable_w)
        if t_obj:
            story.append(t_obj)

    print(f"Building PDF document with {len(story)} flowables...")
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Document successfully created at: {target_pdf_path}")
    print(f"File size: {os.path.getsize(target_pdf_path):,} bytes")


if __name__ == '__main__':
    source_md = os.path.abspath('docs/technical_report.md')
    target_pdf = os.path.abspath('docs/Faizan_Iqbal_Technical_Report.pdf')
    compile_pdf(source_md, target_pdf)
