"""Build an editable, offline Section 11 presentation from saved analysis.

Uses the standard library plus the project's existing Matplotlib/Pillow stack.
No model, evaluator, network access, or scientific execution is performed.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import re
import textwrap
from typing import Any
from xml.etree import ElementTree
from xml.sax.saxutils import escape
import zipfile

WIDTH, HEIGHT = 13.333333, 7.5
NAVY, TEAL, GOLD = "142B43", "177E89", "BA7C26"
INK, MUTED, PAPER = "23384D", "526579", "F5F7FA"
WHITE, PALE = "FFFFFF", "E5EFF2"
METHOD_ORDER = (
    "gpt2_xl_frozen", "gpt2_xl_weighted_lora", "gpt2_xl_uniform_lora",
    "qwen3_4b_frozen", "qwen3_4b_weighted_lora", "uniform_random",
    "training_circuit_retrieval", "conventional_vqe",
)
LABELS = {
    "gpt2_xl_frozen": "GPT-2 XL frozen",
    "gpt2_xl_weighted_lora": "GPT-2 XL weighted",
    "gpt2_xl_uniform_lora": "GPT-2 XL uniform",
    "qwen3_4b_frozen": "Qwen3-4B frozen",
    "qwen3_4b_weighted_lora": "Qwen3-4B weighted",
    "uniform_random": "Uniform random",
    "training_circuit_retrieval": "Training-circuit retrieval",
    "conventional_vqe": "Compact excitation VQE",
}


@dataclass
class Element:
    kind: str
    x: float
    y: float
    w: float
    h: float
    text: str = ""
    size: float = 20
    color: str = INK
    fill: str | None = None
    bold: bool = False
    path: str | None = None


@dataclass
class Slide:
    title: str
    elements: list[Element]
    script: str
    seconds: int = 60


def _text(elements: list[Element], text: str, x: float, y: float, w: float,
          h: float, *, size: float = 20, color: str = INK,
          bold: bool = False) -> None:
    elements.append(Element("text", x, y, w, h, text, size, color, bold=bold))


def _rect(elements: list[Element], x: float, y: float, w: float, h: float,
          fill: str) -> None:
    elements.append(Element("rect", x, y, w, h, fill=fill))


def _base(number: int, title: str, subtitle: str) -> list[Element]:
    e: list[Element] = []
    _rect(e, 0, 0, WIDTH, HEIGHT, WHITE)
    _rect(e, 0, 0, WIDTH, .14, TEAL)
    _text(e, "SC4055  /  CO2 CIRCUIT GENERATION", .55, .36, 11, .25,
          size=11, color=TEAL, bold=True)
    _text(e, title, .55, .84, 12.15, .95, size=28, color=NAVY, bold=True)
    _text(e, subtitle, .55, 1.84, 12.15, .55, size=14, color=MUTED)
    _rect(e, .55, 7.03, 12.23, .012, "D8E0E7")
    _text(e, "One pipeline seed · two fixed targets · classical simulation", .55,
          7.15, 11.4, .22, size=10, color=MUTED)
    _text(e, f"{number:02d} / 15", 12.03, 7.15, .75, .22,
          size=10, color=MUTED)
    return e


def _bullets(e: list[Element], items: list[str], *, x: float = .65,
             y: float = 2.55, w: float = 11.9, size: float = 21,
             gap: float = .35) -> None:
    for item in items:
        # Explicit line breaks keep PDF and PowerPoint wrapping identical.
        width = max(20, int(w * 72 / (size * .52)))
        lines = textwrap.wrap(item, width=width, break_long_words=False,
                              break_on_hyphens=False)
        height = len(lines) * size * 1.25 / 72
        _text(e, "•", x, y, .25, .35, size=size, color=TEAL)
        _text(e, "\n".join(lines), x + .35, y, w - .35, height,
              size=size)
        y += height + gap
    if y - gap > 6.92:
        raise ValueError("Slide bullets exceed the available space")


def _table(e: list[Element], headings: list[str], rows: list[list[str]],
           *, widths: list[float], x: float = .65, y: float = 2.5,
           row_h: float = .43, font: float = 15) -> None:
    _rect(e, x, y, sum(widths), row_h, NAVY)
    for idx, row in enumerate([headings] + rows):
        yy = y + idx * row_h
        if idx:
            _rect(e, x, yy, sum(widths), row_h,
                  PALE if idx % 2 else PAPER)
        xx = x
        for value, width in zip(row, widths):
            _text(e, value, xx + .12, yy + .08, width - .2, row_h - .1,
                  size=font, color=WHITE if idx == 0 else INK,
                  bold=idx == 0)
            xx += width


def _image(e: list[Element], path: Path, *, x: float = .6, y: float = 2.4,
           w: float = 12.1, h: float = 4.4) -> None:
    from PIL import Image
    with Image.open(path) as im:
        aspect = im.width / im.height
    actual_w = min(w, h * aspect)
    actual_h = actual_w / aspect
    e.append(Element("image", x + (w - actual_w) / 2,
                     y + (h - actual_h) / 2, actual_w, actual_h,
                     path=str(path)))


def _emu(inches: float) -> int:
    return round(inches * 914400)


def _shape_xml(element: Element, shape_id: int) -> str:
    x, y, w, h = (_emu(v) for v in (element.x, element.y, element.w, element.h))
    transform = (f'<a:xfrm><a:off x="{x}" y="{y}"/>'
                 f'<a:ext cx="{w}" cy="{h}"/></a:xfrm>')
    fill = (f'<a:solidFill><a:srgbClr val="{element.fill}"/></a:solidFill>'
            if element.fill else '<a:noFill/>')
    paragraphs = []
    for line in element.text.split("\n"):
        paragraphs.append(
            '<a:p><a:pPr marL="0" marR="0" indent="0">'
            '<a:lnSpc><a:spcPct val="100000"/></a:lnSpc>'
            '<a:spcBef><a:spcPts val="0"/></a:spcBef>'
            '<a:spcAft><a:spcPts val="0"/></a:spcAft></a:pPr>'
            f'<a:r><a:rPr lang="en-US" sz="{round(element.size*100)}" '
            f'b="{int(element.bold)}"><a:solidFill>'
            f'<a:srgbClr val="{element.color}"/></a:solidFill>'
            '<a:latin typeface="Aptos"/></a:rPr>'
            f'<a:t xml:space="preserve">{escape(line)}</a:t></a:r>'
            f'<a:endParaRPr lang="en-US" sz="{round(element.size*100)}"/></a:p>'
        )
    body = ('<p:txBody><a:bodyPr wrap="square" lIns="0" tIns="0" '
            'rIns="0" bIns="0" anchor="t"><a:noAutofit/></a:bodyPr>'
            '<a:lstStyle/>' + ''.join(paragraphs) + '</p:txBody>')
    return (
        '<p:sp><p:nvSpPr>'
        f'<p:cNvPr id="{shape_id}" name="Shape {shape_id}"/>'
        f'<p:cNvSpPr txBox="{int(element.kind == "text")}"/><p:nvPr/>'
        '</p:nvSpPr><p:spPr>' + transform +
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>' + fill +
        '<a:ln><a:noFill/></a:ln></p:spPr>' + body + '</p:sp>'
    )


def _picture_xml(element: Element, shape_id: int, relationship_id: str) -> str:
    x, y, w, h = (_emu(v) for v in (element.x, element.y, element.w, element.h))
    return (
        '<p:pic><p:nvPicPr>'
        f'<p:cNvPr id="{shape_id}" name="Evidence figure {shape_id}"/>'
        '<p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr>'
        '<p:nvPr/></p:nvPicPr><p:blipFill>'
        f'<a:blip r:embed="{relationship_id}"/>'
        '<a:stretch><a:fillRect/></a:stretch></p:blipFill><p:spPr>'
        f'<a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{h}"/>'
        '</a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
        '</p:spPr></p:pic>'
    )


P_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
GROUP = ('<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/>'
         '<p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm>'
         '<a:off x="0" y="0"/><a:ext cx="0" cy="0"/>'
         '<a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/>'
         '</a:xfrm></p:grpSpPr>')


def _rels(entries: list[tuple[str, str, str]]) -> str:
    return (f'<Relationships xmlns="{REL_NS}">' + ''.join(
        f'<Relationship Id="{rid}" Type="{R_NS}/{kind}" Target="{target}"/>'
        for rid, kind, target in entries) + '</Relationships>')


def _write_pptx(slides: list[Slide], destination: Path) -> None:
    parts: dict[str, str | bytes] = {}
    overrides = [
        ('/ppt/presentation.xml', 'presentationml.presentation.main+xml'),
        ('/ppt/slideMasters/slideMaster1.xml', 'presentationml.slideMaster+xml'),
        ('/ppt/slideLayouts/slideLayout1.xml', 'presentationml.slideLayout+xml'),
        ('/ppt/theme/theme1.xml', 'theme+xml'),
    ]
    for idx in range(1, len(slides) + 1):
        overrides.append((f'/ppt/slides/slide{idx}.xml', 'presentationml.slide+xml'))
    parts['[Content_Types].xml'] = (
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Default Extension="png" ContentType="image/png"/>' + ''.join(
            f'<Override PartName="{name}" ContentType="application/vnd.openxmlformats-officedocument.{kind}"/>'
            for name, kind in overrides) + '</Types>')
    parts['_rels/.rels'] = _rels([('rId1', 'officeDocument', 'ppt/presentation.xml')])
    parts['ppt/presentation.xml'] = (
        f'<p:presentation xmlns:a="{A_NS}" xmlns:r="{R_NS}" xmlns:p="{P_NS}">'
        '<p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>'
        '<p:sldIdLst>' + ''.join(
            f'<p:sldId id="{255+i}" r:id="rId{i+1}"/>' for i in range(1, len(slides)+1)) +
        '</p:sldIdLst>'
        f'<p:sldSz cx="{_emu(WIDTH)}" cy="{_emu(HEIGHT)}" type="screen16x9"/>'
        '<p:notesSz cx="6858000" cy="9144000"/></p:presentation>')
    parts['ppt/_rels/presentation.xml.rels'] = _rels(
        [('rId1', 'slideMaster', 'slideMasters/slideMaster1.xml')] +
        [(f'rId{i+1}', 'slide', f'slides/slide{i}.xml') for i in range(1, len(slides)+1)])
    parts['ppt/slideMasters/slideMaster1.xml'] = (
        f'<p:sldMaster xmlns:a="{A_NS}" xmlns:r="{R_NS}" xmlns:p="{P_NS}">'
        '<p:cSld><p:spTree>' + GROUP + '</p:spTree></p:cSld>'
        '<p:clrMap accent1="accent1" accent2="accent2" accent3="accent3" '
        'accent4="accent4" accent5="accent5" accent6="accent6" bg1="lt1" '
        'bg2="lt2" folHlink="folHlink" hlink="hlink" tx1="dk1" tx2="dk2"/>'
        '<p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>'
        '<p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles></p:sldMaster>')
    parts['ppt/slideMasters/_rels/slideMaster1.xml.rels'] = _rels([
        ('rId1', 'slideLayout', '../slideLayouts/slideLayout1.xml'),
        ('rId2', 'theme', '../theme/theme1.xml')])
    parts['ppt/slideLayouts/slideLayout1.xml'] = (
        f'<p:sldLayout xmlns:a="{A_NS}" xmlns:r="{R_NS}" xmlns:p="{P_NS}" '
        'type="blank" preserve="1"><p:cSld name="Blank"><p:spTree>' + GROUP +
        '</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>')
    parts['ppt/slideLayouts/_rels/slideLayout1.xml.rels'] = _rels([
        ('rId1', 'slideMaster', '../slideMasters/slideMaster1.xml')])
    scheme = ''.join(f'<a:{name}><a:srgbClr val="{value}"/></a:{name}>'
                     for name, value in [('dk1', INK), ('lt1', WHITE), ('dk2', NAVY),
                                         ('lt2', PAPER), ('accent1', TEAL), ('accent2', GOLD),
                                         ('accent3', '3F76A3'), ('accent4', '944C62'),
                                         ('accent5', '62836F'), ('accent6', '8A73A9'),
                                         ('hlink', TEAL), ('folHlink', '944C62')])
    font = '<a:latin typeface="Aptos"/><a:ea typeface=""/><a:cs typeface=""/>'
    parts['ppt/theme/theme1.xml'] = (
        f'<a:theme xmlns:a="{A_NS}" name="CO2 research"><a:themeElements>'
        '<a:clrScheme name="CO2">' + scheme + '</a:clrScheme>'
        '<a:fontScheme name="CO2"><a:majorFont>' + font + '</a:majorFont>'
        '<a:minorFont>' + font + '</a:minorFont></a:fontScheme>'
        '<a:fmtScheme name="CO2"><a:fillStyleLst>' +
        '<a:solidFill><a:schemeClr val="phClr"/></a:solidFill>'*3 +
        '</a:fillStyleLst><a:lnStyleLst>' +
        '<a:ln w="9525"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:prstDash val="solid"/></a:ln>'*3 +
        '</a:lnStyleLst><a:effectStyleLst>' + '<a:effectStyle><a:effectLst/></a:effectStyle>'*3 +
        '</a:effectStyleLst><a:bgFillStyleLst>' +
        '<a:solidFill><a:schemeClr val="phClr"/></a:solidFill>'*3 +
        '</a:bgFillStyleLst></a:fmtScheme></a:themeElements></a:theme>')
    image_number = 0
    for index, slide in enumerate(slides, 1):
        slide_rels = [('rId1', 'slideLayout', '../slideLayouts/slideLayout1.xml')]
        shapes = []
        for shape_id, element in enumerate(slide.elements, 2):
            if element.kind == 'image':
                image_number += 1
                rid = f'rId{len(slide_rels)+1}'
                target = f'../media/image{image_number}.png'
                slide_rels.append((rid, 'image', target))
                parts[f'ppt/media/image{image_number}.png'] = Path(element.path).read_bytes()
                shapes.append(_picture_xml(element, shape_id, rid))
            else:
                shapes.append(_shape_xml(element, shape_id))
        parts[f'ppt/slides/slide{index}.xml'] = (
            f'<p:sld xmlns:a="{A_NS}" xmlns:r="{R_NS}" xmlns:p="{P_NS}">'
            '<p:cSld><p:spTree>' + GROUP + ''.join(shapes) +
            '</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>')
        parts[f'ppt/slides/_rels/slide{index}.xml.rels'] = _rels(slide_rels)
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, content in parts.items():
            archive.writestr(name, content if isinstance(content, bytes) else
                             ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>' + content).encode())
    # Validate every XML part and relationship before calling this a deliverable.
    import posixpath
    with zipfile.ZipFile(destination) as archive:
        names = set(archive.namelist())
        if archive.testzip() is not None:
            raise ValueError("Presentation ZIP integrity check failed")
        for name in names:
            if name.endswith(('.xml', '.rels')):
                document = ElementTree.fromstring(archive.read(name))
                if name.endswith('.rels'):
                    base = '' if name == '_rels/.rels' else name.split('/_rels/')[0]
                    for relationship in document:
                        target = posixpath.normpath(posixpath.join(base, relationship.attrib['Target']))
                        if target not in names:
                            raise ValueError(f"Missing PPTX relationship target: {target}")


def _write_pdf(slides: list[Slide], output: Path, previews: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Rectangle
    from PIL import Image
    previews.mkdir(parents=True, exist_ok=True)
    with PdfPages(output, metadata={'Title': '30-qubit CO2 circuit generation',
                                   'Subject': 'SC4055 Section 11 presentation'}) as pdf:
        for number, slide in enumerate(slides, 1):
            fig = plt.figure(figsize=(WIDTH, HEIGHT), facecolor='white')
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_xlim(0, WIDTH)
            ax.set_ylim(HEIGHT, 0)
            ax.axis('off')
            for layer, e in enumerate(slide.elements):
                if e.kind == 'rect':
                    ax.add_patch(Rectangle((e.x, e.y), e.w, e.h,
                                           facecolor='#' + str(e.fill), linewidth=0,
                                           zorder=layer))
                elif e.kind == 'image':
                    with Image.open(e.path) as im:
                        ax.imshow(im, extent=[e.x, e.x+e.w, e.y+e.h, e.y],
                                  aspect='auto', zorder=layer)
                else:
                    ax.text(e.x, e.y, e.text, fontsize=e.size, color='#'+e.color,
                            fontfamily='DejaVu Sans', weight='bold' if e.bold else 'normal',
                            va='top', ha='left', linespacing=1.0, zorder=layer)
            pdf.savefig(fig)
            fig.savefig(previews / f'slide_{number:02d}.png', dpi=110)
            plt.close(fig)


def _mmss(seconds: int) -> str:
    return f'{seconds//60:02d}:{seconds%60:02d}'


def _write_script(slides: list[Slide], destination: Path) -> None:
    total = sum(s.seconds for s in slides)
    words = sum(len(re.findall(r"\b[\w]+(?:[-'][\w]+)*\b", s.script)) for s in slides)
    lines = ['# Fifteen-minute recording script', '',
             f'**Schedule:** {_mmss(total)}; {len(slides)} slides. **Spoken script:** {words:,} words.', '',
             'Read at approximately 125–135 words per minute. Use the remaining time in each slot '
             'to point at figures, change slides, and pause. Rehearse once and trim pauses to finish '
             'at 15:00. This is a recording aid; the video remains for the researcher to record.', '',
             'The editable PowerPoint and matching PDF are in this folder. Figures are drawn from '
             'the saved Section 11 analysis; no new inference or energy evaluations are involved.', '']
    current = 0
    for index, slide in enumerate(slides, 1):
        lines += [f'## Slide {index}: {slide.title}', '',
                  f'**{_mmss(current)}–{_mmss(current+slide.seconds)}**', '', slide.script, '']
        current += slide.seconds
    destination.write_text('\n'.join(lines), encoding='utf-8')


def make_slides(root: Path, analysis: dict[str, Any], costs: dict[str, Any]) -> list[Slide]:
    """Use the authenticated matrix report's public keys; calculate no new energies."""
    methods = analysis['methods']
    pairs = analysis['paired_comparisons']
    jobs = analysis['jobs']
    figures = root / 'large_qubit/analysis/section11_v1/figures'
    means = {k: 1000*v['mean_verified_error_hartree'] for k, v in methods.items()}
    gpt_gain = -1000*pairs['gpt_weighted_minus_frozen']['mean_difference_hartree']
    qwen_gain = -1000*pairs['qwen_weighted_minus_frozen']['mean_difference_hartree']
    weighting_gain = -1000*pairs['gpt_weighted_minus_uniform']['mean_difference_hartree']
    total = costs['ordinary_requests']['actual']
    ceiling = costs['ordinary_requests']['amended_ceiling']
    train_hours = costs['training']['assigned_device_hours']
    recovery_hours = next(p['recorded_seconds']/3600 for p in costs['phase_costs']
                          if p['phase'] == 'section9_recovery')
    slides: list[Slide] = []

    def add(title: str, subtitle: str, script: str, bullets: list[str] | None = None,
            figure: str | None = None) -> list[Element]:
        e = _base(len(slides)+1, title, subtitle)
        if bullets:
            _bullets(e, bullets)
        if figure:
            _image(e, figures / figure)
        slides.append(Slide(title, e, ' '.join(script.split())))
        return e

    e = add('Can pretrained models propose useful CO2 circuits?',
            'A fixed-budget, 30-qubit classical-simulation case study',
            "This project asks whether language-pretrained models can propose useful quantum circuits "
            "for molecular energy search after a small amount of task-specific adaptation. The new "
            "benchmark is linear carbon dioxide in the STO-3G basis, encoded on thirty qubits. "
            "We compare eight methods on two untouched interpolation geometries using one complete "
            "pipeline seed. The central finding is mixed but informative. Adaptation greatly improves "
            "both frozen backbones, while the remaining error is substantial. Retrieval from the "
            "same classical training corpus is extremely competitive, and the conventional variational "
            "baseline wins one target. Therefore the contribution is an auditable comparison with "
            "careful accounting, rather than a claim that a language model solves chemistry or "
            "outperforms classical methods generally. I will explain the protocol, results, costs, "
            "and the exact limits of those conclusions.",
            ['Adaptation improves both frozen backbones.',
             'Retrieval and compact VQE remain strong competitors.',
             'All 16 final trajectories and their winning energies are verified.'])

    add('Four questions, one bounded experiment',
        'Seed 22 · 8 methods × 2 targets · allowance 1,024 requests per trajectory',
        "There are four research questions. First, does energy-weighted adaptation improve each "
        "backbone relative to its frozen version? Second, how do these learned proposals compare "
        "with random search, training-circuit retrieval, and compact excitation VQE? Third, do "
        "the fitted models propose useful circuits at the two fixed unseen geometries? Fourth, "
        "does energy weighting help beyond uniform supervised training for GPT-2? We deliberately "
        "omit a Qwen uniform condition, so the weighting-specific comparison is only available "
        "for GPT-2. Every trajectory receives the same ordinary request allowance, including "
        "its initial Hartree-Fock request. Duplicate and cached requests still consume that "
        "allowance. These controls make the case study interpretable, but sixteen trajectories "
        "are not sixteen independent training seeds. The two-target mean summarizes these "
        "two geometries only, not a molecular population.",
        ['RQ1: weighted adapter versus frozen backbone.',
         'RQ2: learned proposals versus retrieval, random search and compact VQE.',
         'RQ3: transfer to R = 1.10 and 1.30 angstrom.',
         'RQ4: weighted versus uniform GPT-2 training.'])

    add('Thirty encoded qubits; a tractable population sector',
        'Linear O–C–O · full STO-3G space · 22 electrons · 15 spatial orbitals',
        "The physical system uses the full STO-3G orbital space, without frozen-core removal "
        "or qubit tapering. Fifteen spatial orbitals give thirty interleaved spin orbitals. "
        "We fix eleven alpha and eleven beta electrons. That population sector has one "
        "million, eight hundred sixty-three thousand, two hundred twenty-five determinants. "
        "The evaluator applies the Hamiltonian without constructing a dense matrix. Its "
        "reference is the lowest energy in the same finite-basis, fixed-population sector. "
        "It is not an experimental molecular energy, and fixed spin projection alone does "
        "not prove total-spin purity. Orbital conventions, transformed integrals and nuclear "
        "constants are bound to saved physical fingerprints. Small-system regression checks "
        "and fresh winner evaluations support numerical correctness. This reduced sector also "
        "explains why a thirty-qubit encoding remains accessible on a classical CPU; width "
        "alone says nothing about quantum advantage.",
        ['N-alpha = N-beta = 11; 1,863,225 determinant amplitudes.',
         'Matrix-free CPU energies and a same-Hamiltonian classical reference.',
         'No hardware execution, total-spin claim or classical-hardness claim.'])

    add('Offline learned proposals and credible controls',
        'Inspired by GQE; the implemented objective is offline weighted likelihood',
        "The learned method is offline energy-weighted supervised LoRA. Frozen pretrained "
        "weights receive low-rank updates from complete circuit answers, with prompts and "
        "padding excluded from the loss. Final generation receives no target-energy feedback "
        "and performs no online model update. That distinguishes this implementation from "
        "the full generative quantum eigensolver algorithms in the literature. Learned "
        "methods, random search and retrieval share a screened, discrete whole-qubit "
        "excitation family with thirty-two positions. Retrieval orders training circuits "
        "using source energies at the nearest training geometry. The VQE comparator uses "
        "a separate compact fermionic excitation family with twenty-four continuous parameters. "
        "These families have different parity semantics and expressivity. Consequently, "
        "the VQE comparison concerns complete methods, not an isolated optimizer effect. "
        "The report retains that distinction throughout the accuracy and resource tables.",
        ['3 fits: GPT-2 weighted, GPT-2 uniform, Qwen weighted.',
         '5 model conditions; random, retrieval and compact VQE complete the matrix.',
         'Discrete proposals: 32 whole-qubit positions. VQE: 24 fermionic parameters.',
         'No target-energy feedback to the generators.'])

    add('Freeze data, fitting and validation before targets',
        'Three training geometries → one shared corpus → three fits → one validation panel',
        "The shared release contains six thousand, one hundred forty-four fitting rows and "
        "three hundred eighty-four separate diagnostics. They come from training distances "
        "of one point zero, one point two, and one point three two angstrom. The first "
        "acquisition campaign failed a prespecified weight-concentration gate; its data were "
        "excluded from the confirmation release, but its costs and negative evidence remain. "
        "An explicitly approved fresh campaign passed the unchanged quality criteria. All "
        "three adapters then completed five corpus passes, using their final checkpoints. "
        "The two GPT-2 conditions share initialization, row order and numerical settings "
        "apart from their objectives. A compact validation panel at one point two five "
        "angstrom preceded the final freeze. Only afterward were the two target references "
        "and main trajectories executed, preserving the intended separation between development "
        "and final scoring.",
        ['6,144 fitting rows + 384 isolated diagnostics; fresh v2 release only.',
         'Five passes per fit; matched GPT-2 initialization, order and numerics.',
         'Validation at R = 1.25 angstrom; final targets withheld until freeze.',
         'Rejected acquisition and interrupted work remain in the cost record.'])

    e = add('All sixteen final endpoints',
            'Verified error above the sector reference, in mHa; lower is better',
            f"This table shows every final result, including both targets and their fixed "
            f"two-target mean. Weighted Qwen has the lowest mean error, {means['qwen3_4b_weighted_lora']:.3f} "
            f"millihartree, followed closely by retrieval at {means['training_circuit_retrieval']:.3f}. "
            "Compact VQE wins the shorter-distance target, whereas retrieval wins the longer-distance "
            "target. Weighted GPT-2 is close to these methods, and uniform GPT-2 is also much "
            "better than its frozen control. Frozen Qwen's best result remains Hartree-Fock "
            "on both targets. These are verified best-of-budget outcomes, not averages over "
            "generated circuits. The mean does not hide the target-specific ranking reversal. "
            "All errors are relative to the same finite-basis Hamiltonian for each target. "
            "The large remaining errors mean that improved generation should not be mistaken "
            "for reaching chemically accurate energies or solving the molecular ground-state "
            "problem to a demanding tolerance.")
    rows = [[LABELS[m], f"{1000*methods[m]['target_errors_hartree']['CO2_I01']:.3f}",
             f"{1000*methods[m]['target_errors_hartree']['CO2_I02']:.3f}", f"{means[m]:.3f}"]
            for m in METHOD_ORDER]
    _table(e, ['Method', 'I01: R = 1.10', 'I02: R = 1.30', 'Fixed-target mean'],
           rows, widths=[5.1, 2.2, 2.2, 2.35], row_h=.43, font=15)
    _text(e, 'Reference error = 0 by definition; the endpoint is best verified E − E_reference.',
          .75, 6.58, 11.7, .25, size=12, color=MUTED)

    add('Adaptation is the large effect; weighting is smaller',
        'Paired differences use the same target; signs and units are explicit in the figure',
        f"The paired comparisons separate two effects. Relative to the frozen backbone, "
        f"weighted GPT-2 reduces mean error by {gpt_gain:.3f} millihartree, and weighted "
        f"Qwen reduces it by {qwen_gain:.3f}. Both improvements occur on both targets. "
        f"The weighting-specific GPT-2 effect is much smaller: {weighting_gain:.3f} "
        "millihartree averaged over the two targets. At the first target it is only "
        "about zero point zero zero six millihartree, while at the second it is about "
        "one point three millihartree. Therefore the broad adaptation improvement should "
        "not be attributed entirely to energy weighting. We cannot perform a corresponding "
        "Qwen weighting comparison because no Qwen uniform fit was run. Nor do these "
        "few paired outcomes justify significance tests or training-seed confidence intervals. "
        "They are precise numerical differences for this frozen experimental design.",
        figure='paired_effects.png')

    add('Read the complete request curves',
        'Every logical request is charged, including duplicates and cache hits',
        "The request curves show how the best incumbent develops across the full allowance. "
        "They preserve early plateaus, late improvements and target-dependent behavior that "
        "an endpoint table cannot show. The horizontal axis counts logical requests, not "
        "only expensive cache misses. This prevents repeated circuits from receiving free "
        "search opportunities. Hartree-Fock provides the initial incumbent, and the sector "
        "reference supplies the common zero-error context. The two compact VQE trajectories "
        "each use one fewer request than the maximum because their frozen optimizer "
        "schedule cannot use the final unit; the incumbent is carried to the common "
        "endpoint explicitly. These curves describe one realized search path per method "
        "and target. They do not show uncertainty across seeds, and the individual points "
        "must not be resampled as if they were independent trained models.",
        figure='budget_curves.png')

    add('Validity does not guarantee useful or diverse search',
        'Duplicates remain charged; the selected HF circuit has zero post-HF gate cost',
        "Constrained decoding successfully enforces the circuit grammar, but grammatical "
        "validity is not physical usefulness or diversity. Frozen Qwen repeats almost "
        "entirely: each target records one thousand twenty-two duplicate requests, and "
        "its verified winner stays at Hartree-Fock. Weighted GPT-2 also has many duplicates, "
        "more than uniform GPT-2, despite slightly better final energies. The ledger "
        "keeps these requests instead of discarding them from the denominator. The resource "
        "panel describes the selected winners under a common logical decomposition. A "
        "Hartree-Fock winner has zero additional gates after state preparation, which is "
        "a resource fact rather than evidence of useful correlated search. Neither a "
        "valid string nor a low gate count is sufficient by itself. Accuracy, diversity, "
        "logical request consumption and circuit resources must be considered together.",
        figure='duplicates_resources.png')

    e = add('Count acquisition, failures and fitting work',
            'Logical circuit-energy requests; overhead and device time are reported separately',
            f"The complete new-branch ordinary request count is {total:,}, within the "
            f"amended ceiling of {ceiling:,}. This exceeds the original planning allowance "
            "because approved amendments retained the rejected acquisition campaign and the "
            "failed first main attempt. Hiding either would understate research expenditure. "
            "References and cache-independent winner verification remain separate overhead because their "
            "Hamiltonian actions are not interchangeable with ordinary circuit requests. "
            f"The recovered four-lane matrix took about {recovery_hours:.2f} elapsed hours. "
            f"The three fits record at least {train_hours:.2f} assigned-device hours, including "
            "attributable replay and loading; interrupted records make that a lower bound. "
            "Those device hours cannot be added to overlapping component timings as if "
            "they were elapsed project time. Marginal search cost also excludes shared "
            "data acquisition, so it does not establish an amortized transfer advantage.")
    display = {'training_only_pilot': 'Training-only pilots',
               'acquisition_including_rejected_v1': 'Corpus: rejected v1 + fresh v2',
               'validation': 'Compact validation panel',
               'main_including_failed_attempt': 'Main: failed + recovery attempts'}
    cr = [[display[r['category']], f"{r['actual']:,}", f"{r['ceiling']:,}"]
          for r in costs['ordinary_requests']['categories']]
    cr.append(['Total ordinary requests', f'{total:,}', f'{ceiling:,}'])
    _table(e, ['Category', 'Actual', 'Amended ceiling'], cr,
           widths=[7.05, 2.4, 2.4], row_h=.55, font=16)
    _text(e, f'Main recovery: {recovery_hours:.2f} h elapsed  |  Fits: ≥ {train_hours:.2f} assigned-device h',
          .75, 6.15, 11.7, .4, size=18, color=TEAL, bold=True)
    _text(e, 'References, verification and historical LiH are separate. Timing scopes differ.',
          .75, 6.65, 11.7, .25, size=12, color=MUTED)

    add('Circuit resources are logical implementation costs',
        'Common gate set and conventions; different native families remain explicit',
        "Circuit-resource reporting uses an explicit decomposition into common single-qubit "
        "gates and controlled-NOT gates. The saved sequence, count and dependency depth "
        "can be inspected for every winning circuit. These are logical all-to-all "
        "resources, with no hardware routing, native-device optimization or noise model. "
        "Headline counts exclude the twenty-two bit flips that prepare the Hartree-Fock "
        "occupation; inclusive counts are saved as well. The decomposition is exact for "
        "the implemented generators, without a Trotter approximation, but it preserves "
        "the difference between whole-qubit and fermionic excitation semantics. A smaller "
        "reported depth therefore does not establish superiority on a physical processor. "
        "Likewise, touching thirty wires does not show that all encoded degrees of "
        "freedom contribute meaningfully to correlation. The scientific comparison remains "
        "accuracy and measured cost for the specified complete methods, with these resource "
        "conventions stated alongside it.",
        ['Saved common gate sequences, CX counts and dependency depths for all winners.',
         'All-to-all logical connectivity; no routing, noise or hardware optimization.',
         'Headline costs exclude HF preparation; inclusive counts are also retained.',
         'Whole-qubit proposals and fermionic VQE remain different families.'])

    add('Retained LiH evidence has its own scope',
        'Historical 12-qubit pilot/development data; never pooled with CO2 endpoints',
        "The earlier LiH work remains useful historical evidence, but it has a different "
        "scope. Its pilot used seed eleven and training geometries, so it is not an "
        "additional held-out CO2 replication. Weighted GPT-2 generated lower mean energy "
        "than uniform GPT-2 despite worse diagnostic likelihood on uniform strings. That "
        "already warned that likelihood and energy answer different questions. The historical "
        "acquisition also showed that classically guided local proposals supplied almost "
        "all strongly improving examples, making source-data cost central to interpretation. "
        "The original LiH development convergence gate remains false; completing CO2 does "
        "not retroactively change it. Its datasets, checkpoints, figures, registrations "
        "and cost records are retained under their original identity. The report links "
        "those artifacts and provides a separate historical panel, rather than relabelling "
        "them as new successes or mixing their expenditures into the CO2 main matrix.",
        ['Seed-11 pilot: training-geometry evidence, not new held-out results.',
         'Weighted GPT-2: lower generated energy despite worse uniform-string NLL.',
         'Classical acquisition supplied useful examples; its costs stay visible.',
         'The original LiH convergence gate remains false.'])

    add('What the experiment cannot establish',
        'Numerical correctness and successful execution do not imply broad superiority',
        "Several limits are essential to the conclusion. One pipeline seed cannot measure "
        "typical-seed reliability, and two fixed interpolation geometries do not establish "
        "transfer across molecules or extrapolation beyond the training range. The backbones "
        "differ in size, tokenizer, architecture and pretraining, so their comparison does "
        "not isolate the causal value of language pretraining. We have no Qwen uniform "
        "control, no matched-family angle-optimized comparator, and no learned-generator warm-start VQE "
        "condition. Therefore weighting-specific, optimizer-specific and amortized-transfer "
        "claims must remain correspondingly narrow. The remaining reference errors are "
        "large, and simulation of a reduced population sector proves neither noisy-device "
        "performance nor quantum advantage. Future work should address these missing controls "
        "and independent seeds before increasing system width. More qubits alone would "
        "not resolve the central uncertainty about comparative usefulness and total cost.",
        ['No seed-level confidence interval or significance claim.',
         'No general cross-molecule or extrapolation claim.',
         'No isolated pretraining, optimizer or amortized-transfer advantage.',
         'No chemical-accuracy, noisy-device or quantum-advantage claim.'])

    add('A reproducible result, inspectable offline',
        'Brief recording demo: show saved artifacts and the verification receipt',
        "For the reproducibility demonstration, I start in the submission folder and open "
        "the endpoint table and the paired-effect figure. These come from saved authenticated "
        "records, not a live model call. Next I show the matrix verification receipt: "
        "all sixteen trajectories are accounted for, the matrix is complete, and every "
        "trajectory is successful. The failed first attempt remains separately charged "
        "in the ordinary accounting. Finally, I point to the frozen protocol, physical "
        "fingerprints, shared corpus release, final adapters and per-trajectory ledgers. "
        "The documented Linux workflow can reproduce execution, while this analysis and "
        "presentation can be regenerated locally from saved results. The demo deliberately "
        "uses file inspection so that the recording needs no GPU, network connection "
        "or new energy evaluation. Exact hashes connect the report's claims to the "
        "preserved evidence rather than to screenshots alone.",
        ['Open the saved endpoint table and paired-effect figure.',
         'Show all_passed = true and all_trajectories_successful = true.',
         'Trace results → ledgers → final freeze → physical fingerprints.',
         'Use offline_demo.md; no GPU or new scientific execution required.'])

    add('Useful adaptation; strong classical competition',
        'The evidence supports a descriptive transfer case study, with mixed comparative results',
        "Returning to the research questions, weighted adaptation improves each frozen "
        "backbone at both fixed targets. Learned proposals also outperform uniform random "
        "search in final error, but retrieval and compact VQE remain strong competitors. "
        "The two interpolation results show useful transfer within this one molecular "
        "family and this frozen setup. GPT-2 energy weighting yields a small additional "
        "improvement over uniform fitting, much smaller than the adaptation effect itself. "
        "The practical lesson is that classical acquisition, grammar constraints and "
        "task-specific adaptation jointly shape the result; the pretrained model cannot "
        "take sole credit. The project delivers an authenticated benchmark, complete "
        "trajectories, explicit failure accounting, reproducible figures and restrained "
        "claims. The next scientific priority is independent replication and stronger "
        "matched controls, rather than a larger qubit count alone. Thank you for "
        "watching; the report and documented artifacts contain the detailed evidence.",
        ['Large improvement over frozen controls; small GPT-2 weighting increment.',
         'Retrieval nearly matches the best mean; compact VQE wins one target.',
         'Retain negative outcomes and full costs when interpreting transfer.',
         'Next: independent seeds and matched controls before wider circuits.'])
    if len(slides) != 15:
        raise ValueError('The recording plan requires exactly fifteen slides')
    return slides


def build_presentation(root: Path, analysis: dict[str, Any], costs: dict[str, Any],
                       output_directory: Path) -> list[Path]:
    root, out = Path(root).resolve(), Path(output_directory).resolve()
    out.mkdir(parents=True, exist_ok=True)
    slides = make_slides(root, analysis, costs)
    pptx = out / 'presentation.pptx'
    pdf = out / 'presentation.pdf'
    script = out / 'recording_script.md'
    _write_pptx(slides, pptx)
    _write_pdf(slides, pdf, out / 'previews')
    _write_script(slides, script)
    source = out / 'slides.json'
    portable_slides = [asdict(s) for s in slides]
    for slide in portable_slides:
        for element in slide['elements']:
            if element['path']:
                element['path'] = Path(element['path']).relative_to(root).as_posix()
    source.write_text(json.dumps({'schema_version': 'section11-presentation-v1',
                                 'total_seconds': sum(s.seconds for s in slides),
                                 'image_paths_relative_to_project_root': True,
                                 'slides': portable_slides}, indent=2), encoding='utf-8')
    demo = out / 'offline_demo.md'
    demo.write_text('''# Offline recording demo

Use the 13:00–14:00 slot on slide 14. Before recording, open the following files so the
demo consists of switching windows. No scientific execution is required.

1. **13:00–13:15:** Show slide 6's complete endpoint table (in `presentation.pdf`).
2. **13:15–13:30:** Open `large_qubit/analysis/section11_v1/figures/paired_effects.png`.
   Point to the large adaptation effects and the much smaller weighting effect.
3. **13:30–13:45:** Open the saved verification receipt:
   `large_qubit/evidence/server_retirement_audit/20261108T141756Z/section9_complete_closure.json`.
   Show its `all_passed` and completion/success flags and 16,398 cumulative requests,
   including the 16 failed first-attempt requests. Its completion SHA is also bound by
   the Section 11 analysis. Showing a saved receipt is not a live re-verification.
4. **13:45–14:00:** Show the per-trajectory ledgers and `large_qubit/freezes/section8_v1/freeze.json`,
   then return to the final slide. Full authentication commands are in the submission's
   reproduction guide; do not wait for a long hash scan during the recorded minute.

Run no training, model loading or circuit evaluation during this demonstration.
Use the narration in `recording_script.md`; pause briefly while switching windows.

## Recording checklist

- Keep this sealed `section11_v1` package unchanged. Copy `presentation.pptx` and
  `recording_script.md` into a separate working folder, for example
  `large_qubit/submission/course_final/`, before making presentation edits.
- Add the presenter name/student ID to the working copy's first slide if required by the course.
- Record all 15 slides and the brief demo, with intelligible microphone audio.
- Rehearse against the 60-second slide slots; aim for 15:00 total.
- Export an MP4 using your chosen screen recorder or PowerPoint recording workflow.
  Save the video outside the sealed `section11_v1` folder, for example
  `large_qubit/submission/course_final/presentation_recording.mp4`.
- Check the final video's audio, timing and readable figures before submission.
- The final recorded video is researcher-owned and is not generated by this analysis package.
''', encoding='utf-8')
    validation = out / 'presentation_checks.json'
    with zipfile.ZipFile(pptx) as archive:
        slide_parts = [n for n in archive.namelist() if re.fullmatch(r'ppt/slides/slide\d+\.xml', n)]
        media_parts = [n for n in archive.namelist() if n.startswith('ppt/media/')]
    page_count = len(re.findall(rb'/Type\s*/Page\b', pdf.read_bytes()))
    word_count = sum(len(re.findall(r"\b[\w]+(?:[-'][\w]+)*\b", s.script)) for s in slides)
    if len(slide_parts) != 15 or page_count != 15 or len(media_parts) != 3:
        raise ValueError('Presentation page or figure count is incomplete')
    if not 1800 <= word_count <= 2000 or sum(s.seconds for s in slides) != 900:
        raise ValueError('Presentation narration falls outside its timing plan')
    validation.write_text(json.dumps({
        'schema_version': 'section11-presentation-checks-v1',
        'structural_checks_passed': True,
        'pptx_slides': len(slide_parts), 'pdf_pages': page_count,
        'embedded_evidence_figures': len(media_parts),
        'spoken_script_words': word_count, 'planned_duration_seconds': 900,
        'pptx_xml_and_relationship_targets_checked': True,
        'editable_text_and_table_shapes': True,
        'recorded_video_created': False,
        'native_powerpoint_application_opened': False,
        'native_application_check_note': 'ZIP integrity, XML, relationship targets and PDF pages are checked; native application opening is outside this renderer.',
        'rehearsal_required_for_actual_duration': True,
    }, indent=2), encoding='utf-8')
    return [pptx, pdf, script, source, demo, validation] + sorted((out / 'previews').glob('*.png'))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--analysis', type=Path,
                        default=Path('large_qubit/analysis/section11_v1/analysis.json'))
    parser.add_argument('--costs', type=Path,
                        default=Path('large_qubit/analysis/section11_v1/costs.json'))
    parser.add_argument('--output', type=Path,
                        default=Path('large_qubit/submission/section11_v1/presentation'))
    args = parser.parse_args()
    root = args.root.resolve()
    analysis = json.loads((root / args.analysis).read_text(encoding='utf-8'))
    costs = json.loads((root / args.costs).read_text(encoding='utf-8'))
    artifacts = build_presentation(root, analysis, costs, root / args.output)
    print(json.dumps({'presentation_prepared': True, 'recorded_video_created': False,
                      'artifacts': [str(p.relative_to(root)) for p in artifacts]}, indent=2))


if __name__ == '__main__':
    main()

