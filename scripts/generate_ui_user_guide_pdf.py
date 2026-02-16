"""
Generate annotated UI User Guide PDF from screenshots.
Uses Pillow for image annotation (numbered callouts) and reportlab for PDF.

Usage:
    python scripts/generate_ui_user_guide_pdf.py
"""

import os
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    KeepTogether, Image as RLImage,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
SCREENSHOTS_DIR = BASE_DIR / "docs" / "screenshoots"
ANNOTATED_DIR = BASE_DIR / "docs" / "annotated"
OUTPUT_PDF = BASE_DIR / "docs" / "UI_User_Guide.pdf"

# ---------------------------------------------------------------------------
# Fonts (Windows)
# ---------------------------------------------------------------------------
FONT_PATHS = {
    "regular": [
        r"C:\Windows\Fonts\calibri.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
    ],
    "bold": [
        r"C:\Windows\Fonts\calibrib.ttf",
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\segoeuib.ttf",
    ],
}


def find_font(kind="regular"):
    for p in FONT_PATHS[kind]:
        if os.path.exists(p):
            return p
    raise RuntimeError(f"No {kind} TTF font found for Cyrillic")


FONT_REGULAR_PATH = find_font("regular")
FONT_BOLD_PATH = find_font("bold")

# Register fonts for reportlab
pdfmetrics.registerFont(TTFont("GuideFont", FONT_REGULAR_PATH))
pdfmetrics.registerFont(TTFont("GuideFontBold", FONT_BOLD_PATH))

# Pillow fonts
_pil_font_cache = {}


def pil_font(size, bold=False):
    key = (size, bold)
    if key not in _pil_font_cache:
        path = FONT_BOLD_PATH if bold else FONT_REGULAR_PATH
        _pil_font_cache[key] = ImageFont.truetype(path, size)
    return _pil_font_cache[key]


# ---------------------------------------------------------------------------
# Reportlab styles
# ---------------------------------------------------------------------------
_base_styles = getSampleStyleSheet()

STYLES = {
    "title": ParagraphStyle(
        "Title_RU", parent=_base_styles["Title"],
        fontName="GuideFontBold", fontSize=22, leading=28,
        alignment=TA_CENTER, spaceAfter=6,
    ),
    "subtitle": ParagraphStyle(
        "Subtitle_RU", parent=_base_styles["Normal"],
        fontName="GuideFont", fontSize=12, leading=16,
        alignment=TA_CENTER, textColor=HexColor("#666666"),
        spaceAfter=20,
    ),
    "h1": ParagraphStyle(
        "H1_RU", parent=_base_styles["Heading1"],
        fontName="GuideFontBold", fontSize=18, leading=24,
        spaceBefore=16, spaceAfter=8,
        textColor=HexColor("#1a1a2e"),
    ),
    "h2": ParagraphStyle(
        "H2_RU", parent=_base_styles["Heading2"],
        fontName="GuideFontBold", fontSize=14, leading=18,
        spaceBefore=12, spaceAfter=6,
        textColor=HexColor("#16213e"),
    ),
    "body": ParagraphStyle(
        "Body_RU", parent=_base_styles["Normal"],
        fontName="GuideFont", fontSize=11, leading=15,
        spaceAfter=4,
    ),
    "legend": ParagraphStyle(
        "Legend_RU", parent=_base_styles["Normal"],
        fontName="GuideFont", fontSize=10, leading=14,
        leftIndent=12, spaceAfter=2,
        textColor=HexColor("#333333"),
    ),
    "tip": ParagraphStyle(
        "Tip_RU", parent=_base_styles["Normal"],
        fontName="GuideFont", fontSize=10, leading=13,
        leftIndent=12, spaceAfter=6,
        textColor=HexColor("#0f3460"), backColor=HexColor("#e8f0fe"),
        borderPadding=(4, 6, 4, 6),
    ),
}


# ---------------------------------------------------------------------------
# Image annotation helpers
# ---------------------------------------------------------------------------
CALLOUT_RADIUS = 18
CALLOUT_COLOR = (233, 69, 96)        # #e94560 (app accent)
CALLOUT_TEXT_COLOR = (255, 255, 255)
CALLOUT_FONT_SIZE = 22
ARROW_COLOR = (233, 69, 96, 200)
ARROW_WIDTH = 3


def draw_callout(draw, cx, cy, number, radius=CALLOUT_RADIUS):
    """Draw a filled circle with a number inside."""
    x0, y0 = cx - radius, cy - radius
    x1, y1 = cx + radius, cy + radius
    # Shadow
    draw.ellipse([x0 + 2, y0 + 2, x1 + 2, y1 + 2], fill=(0, 0, 0, 120))
    # Main circle
    draw.ellipse([x0, y0, x1, y1], fill=CALLOUT_COLOR, outline=(255, 255, 255), width=2)
    # Number
    font = pil_font(CALLOUT_FONT_SIZE, bold=True)
    text = str(number)
    bbox = font.getbbox(text)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text((cx - tw // 2, cy - th // 2 - 2), text, fill=CALLOUT_TEXT_COLOR, font=font)


def draw_arrow(draw, x1, y1, x2, y2, color=ARROW_COLOR, width=ARROW_WIDTH):
    """Draw a line with an arrowhead from (x1,y1) to (x2,y2)."""
    draw.line([(x1, y1), (x2, y2)], fill=color, width=width)
    # Arrowhead
    angle = math.atan2(y2 - y1, x2 - x1)
    head_len = 14
    head_angle = math.pi / 6
    for sign in (-1, 1):
        a = angle + math.pi + sign * head_angle
        hx = x2 + head_len * math.cos(a)
        hy = y2 + head_len * math.sin(a)
        draw.line([(x2, y2), (int(hx), int(hy))], fill=color, width=width)


def draw_label(draw, x, y, text, font_size=16, bg_color=(22, 33, 62, 200),
               text_color=(255, 255, 255)):
    """Draw text with semi-transparent background."""
    font = pil_font(font_size)
    bbox = font.getbbox(text)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = 6
    draw.rectangle(
        [x - pad, y - pad, x + tw + pad, y + th + pad],
        fill=bg_color,
    )
    draw.text((x, y), text, fill=text_color, font=font)


def annotate_image(img_path, callouts, crop=None):
    """
    Open image, optionally crop, draw callouts.
    callouts: list of (number, cx, cy) or (number, cx, cy, arrow_to_x, arrow_to_y)
    crop: (left, top, right, bottom) or None
    Returns PIL Image (RGBA).
    """
    img = Image.open(img_path).convert("RGBA")
    if crop:
        img = img.crop(crop)
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for item in callouts:
        num, cx, cy = item[0], item[1], item[2]
        if len(item) >= 5:
            ax, ay = item[3], item[4]
            draw_arrow(draw, cx, cy, ax, ay)
        draw_callout(draw, cx, cy, num)
    img = Image.alpha_composite(img, overlay)
    return img.convert("RGB")


def save_annotated(img, name):
    """Save annotated image and return path."""
    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
    path = ANNOTATED_DIR / f"{name}.jpg"
    img.save(str(path), "JPEG", quality=88)
    return str(path)


def screenshot(n):
    """Return path to screenshot N."""
    return str(SCREENSHOTS_DIR / f"photo_{n}_2026-02-14_20-46-37.jpg")


# ---------------------------------------------------------------------------
# Annotation definitions per screenshot
# ---------------------------------------------------------------------------

def annotate_map_overview():
    """photo_1: Map with clusters and toolbar."""
    img = annotate_image(screenshot(1), [
        # (number, callout_x, callout_y, arrow_target_x, arrow_target_y)
        (1, 80, 80, 38, 48),       # Hamburger menu
        (2, 170, 80, 95, 48),      # Search icon
        (3, 230, 80, 155, 48),     # Map icon
        (4, 275, 80, 210, 48),     # Albums icon
        (5, 1190, 80, 1235, 48),   # Fullscreen button
        (6, 550, 340, 620, 395),   # Cluster with thumbnail
        (7, 1200, 135, 1240, 108), # Layer switcher
    ])
    return save_annotated(img, "01_map_overview")


def annotate_map_filters():
    """photo_2: Map filter drawer."""
    img = annotate_image(screenshot(2), [
        (1, 200, 130, 120, 130),    # Format section header
        (2, 200, 280, 120, 222),    # JPG checkbox
        (3, 200, 470, 68, 470),     # Person filter button
        (4, 200, 600, 95, 600),     # Date From
        (5, 200, 665, 95, 665),     # Date To
        (6, 295, 730, 180, 730),    # Stats line
        (7, 365, 57, 332, 57),      # Close button
    ])
    return save_annotated(img, "02_map_filters")


def annotate_map_persons():
    """photo_3: Map person selector."""
    img = annotate_image(screenshot(3), [
        (1, 260, 470, 200, 470),    # Person button with count "1"
        (2, 280, 565, 200, 565),    # Search input
        (3, 300, 690, 355, 690),    # Selected person (checkmark)
        (4, 300, 630, 65, 630),     # Face thumbnail
    ])
    return save_annotated(img, "03_map_persons")


def annotate_map_dates():
    """photo_4: Map dates + person selected."""
    img = annotate_image(screenshot(4), [
        (1, 80, 487, 68, 487),      # Person badge "1"
        (2, 270, 600, 195, 600),    # Date from filled
        (3, 270, 665, 195, 665),    # Date to filled
    ])
    return save_annotated(img, "04_map_dates")


def annotate_search_dialog():
    """photo_5: Search dialog (iPad)."""
    # Crop: (370, 120, 910, 460) → 540x340 px
    # All coordinates below are relative to the cropped image
    img = annotate_image(screenshot(5), [
        (1, 60, 110, 120, 148),     # Text query field "В горах"
        (2, 60, 195, 120, 225),     # ID field
        (3, 110, 310, 150, 292),    # Cancel button
        (4, 420, 310, 382, 292),    # Find button
    ], crop=(370, 120, 910, 460))
    return save_annotated(img, "05_search_dialog")


def annotate_search_filters():
    """photo_6: Search filter drawer with 2 persons selected."""
    img = annotate_image(screenshot(6), [
        (1, 250, 130, 120, 130),    # Format section
        (2, 200, 487, 82, 487),     # Person button with "2"
        (3, 170, 542, 72, 540),     # Person chip "Аврора"
        (4, 270, 542, 165, 540),    # Person chip "Аэлита"
    ])
    return save_annotated(img, "06_search_filters")


def annotate_search_result():
    """photo_7: Search result with badges."""
    img = annotate_image(screenshot(7), [
        (1, 80, 80, 38, 46),        # Hamburger menu
        (2, 280, 100, 176, 112),    # GPS badge on photo
        (3, 280, 145, 222, 112),    # Format badge HEIC
        (4, 1050, 80, 1110, 46),    # Select button
        (5, 1180, 80, 1185, 46),    # Search similar button
        (6, 1255, 80, 1241, 46),    # Settings button
    ])
    return save_annotated(img, "07_search_result")


def annotate_lightbox():
    """photo_8: Lightbox full view."""
    img = annotate_image(screenshot(8), [
        (1, 975, 65, 1042, 65),     # GPS button
        (2, 1040, 110, 1097, 65),   # Album button
        (3, 1110, 110, 1152, 65),   # Faces button
        (4, 1260, 35, 1228, 72),    # Close button
        (5, 43, 430, 43, 430),      # Left arrow
        (6, 1240, 430, 1240, 430),  # Right arrow
        (7, 640, 850, 640, 808),    # Status bar (filename, match, faces)
    ])
    return save_annotated(img, "08_lightbox")


def annotate_album_picker():
    """photo_9: Album picker modal."""
    img = annotate_image(screenshot(9), [
        (1, 370, 210, 640, 210),    # Search albums field
        (2, 370, 275, 640, 275),    # Create new album button
        (3, 370, 365, 445, 365),    # Album thumbnail
        (4, 910, 140, 860, 140),    # Close button
    ])
    return save_annotated(img, "09_album_picker")


def annotate_face_assign():
    """photo_10: Face detection + assignment."""
    img = annotate_image(screenshot(10), [
        (1, 500, 340, 575, 420),    # Face bounding boxes (green)
        (2, 680, 340, 640, 420),    # Face bounding box (red = selected)
        (3, 820, 470, 820, 470),    # Assignment dropdown
        (4, 780, 415, 810, 418),    # Confidence %
        (5, 930, 580, 975, 580),    # Save button
    ])
    return save_annotated(img, "10_face_assign")


def annotate_albums_page():
    """photo_11: Albums page (desktop shows all features)."""
    img = annotate_image(screenshot(11), [
        (1, 40, 80, 40, 45),        # Nav: Search
        (2, 120, 80, 120, 45),      # Nav: Map
        (3, 190, 80, 190, 45),      # Nav: Albums (active)
        (4, 860, 80, 930, 48),      # Search albums input
        (5, 1200, 80, 1160, 48),    # New album button
        (6, 170, 240, 170, 180),    # Album card
    ])
    return save_annotated(img, "11_albums_page")


# ---------------------------------------------------------------------------
# PDF Builder
# ---------------------------------------------------------------------------

PAGE_W, PAGE_H = A4  # 595 x 842 points
CONTENT_W = PAGE_W - 40 * mm  # usable width


def img_flowable(path, max_width=CONTENT_W, max_height=200 * mm):
    """Create a reportlab Image scaled to fit."""
    img = Image.open(path)
    w, h = img.size
    ratio = min(max_width / w, max_height / h)
    return RLImage(path, width=w * ratio, height=h * ratio)


def legend_items(items):
    """Build legend paragraphs from list of (number, text)."""
    result = []
    for num, text in items:
        result.append(Paragraph(
            f'<b><font color="#e94560">{num}</font></b> — {text}',
            STYLES["legend"],
        ))
    return result


def tip_block(text):
    return Paragraph(f'<i>{text}</i>', STYLES["tip"])


def build_pdf():
    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

    # --- Generate all annotated images ---
    print("Generating annotated screenshots...")
    img_paths = {}
    generators = {
        "map_overview": annotate_map_overview,
        "map_filters": annotate_map_filters,
        "map_persons": annotate_map_persons,
        "map_dates": annotate_map_dates,
        "search_dialog": annotate_search_dialog,
        "search_filters": annotate_search_filters,
        "search_result": annotate_search_result,
        "lightbox": annotate_lightbox,
        "album_picker": annotate_album_picker,
        "face_assign": annotate_face_assign,
        "albums_page": annotate_albums_page,
    }
    for name, fn in generators.items():
        print(f"  {name}...")
        img_paths[name] = fn()

    # --- Build PDF ---
    print("Building PDF...")
    doc = SimpleDocTemplate(
        str(OUTPUT_PDF), pagesize=A4,
        rightMargin=20 * mm, leftMargin=20 * mm,
        topMargin=18 * mm, bottomMargin=18 * mm,
    )
    story = []

    # ===== TITLE PAGE =====
    story.append(Spacer(1, 60))
    story.append(Paragraph("Smart Photo Indexing", STYLES["title"]))
    story.append(Paragraph("Руководство пользователя", STYLES["title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "Мобильный интерфейс: поиск, карта, просмотр, альбомы",
        STYLES["subtitle"],
    ))
    story.append(Spacer(1, 30))
    story.append(Paragraph(
        "Данное руководство описывает основные функции веб-интерфейса, "
        "доступные на мобильных устройствах и планшетах. "
        "Административные функции (индексация, GPS-привязка) не рассматриваются.",
        STYLES["body"],
    ))
    story.append(PageBreak())

    # ===== 1. НАВИГАЦИЯ =====
    story.append(Paragraph("1. Навигация", STYLES["h1"]))
    story.append(Paragraph(
        "На всех страницах вверху расположена панель навигации. "
        "На мобильных устройствах часть элементов управления перенесена в выдвижные панели (drawers), "
        "доступные через кнопку-гамбургер.",
        STYLES["body"],
    ))
    story.append(Spacer(1, 6))
    story.append(img_flowable(img_paths["search_result"]))
    story.extend(legend_items([
        (1, "Меню фильтров (гамбургер) — открывает панель с фильтрами по формату, персонам и датам"),
        (2, "GPS-метка — фото имеет координаты"),
        (3, "Формат файла (HEIC, JPG, NEF и др.)"),
        (4, "Выбрать — режим множественного выделения фотографий"),
        (5, "Поиск похожих — загрузить фото для поиска по изображению"),
        (6, "Настройки — размер плиток и другие параметры отображения"),
    ]))
    story.append(Spacer(1, 8))
    story.append(tip_block(
        "Совет: На планшете панель фильтров показывается слева постоянно. "
        "На телефоне — открывается по нажатию на гамбургер."
    ))
    story.append(PageBreak())

    # ===== 2. ПОИСК =====
    story.append(Paragraph("2. Поиск фотографий", STYLES["h1"]))

    # 2.1 Search dialog
    story.append(Paragraph("2.1. Диалог поиска", STYLES["h2"]))
    story.append(Paragraph(
        "Нажмите на иконку лупы в панели навигации. Откроется диалог поиска.",
        STYLES["body"],
    ))
    story.append(Spacer(1, 4))
    story.append(img_flowable(img_paths["search_dialog"], max_height=100 * mm))
    story.extend(legend_items([
        (1, "Поле текстового запроса — введите описание на русском или английском "
            "(например, «в горах», «дети на пляже», «закат»)"),
        (2, "Поле ID фото — для быстрого перехода к конкретной фотографии по номеру"),
        (3, "Отмена — закрыть диалог"),
        (4, "Найти — выполнить поиск"),
    ]))
    story.append(Spacer(1, 8))
    story.append(tip_block(
        "Совет: Поиск использует нейросеть SigLIP, которая понимает русский язык. "
        "Описывайте что изображено на фото своими словами."
    ))
    story.append(Spacer(1, 10))

    # 2.2 Filters
    story.append(Paragraph("2.2. Фильтры поиска", STYLES["h2"]))
    story.append(Paragraph(
        "Откройте панель фильтров нажатием на гамбургер-меню (три полоски). "
        "Все фильтры применяются мгновенно.",
        STYLES["body"],
    ))
    story.append(Spacer(1, 4))
    story.append(img_flowable(img_paths["search_filters"], max_height=150 * mm))
    story.extend(legend_items([
        (1, "Фильтр по формату — выберите нужные форматы файлов (JPG, HEIC, PNG, NEF)"),
        (2, "Кнопка фильтра по персонам — число показывает количество выбранных"),
        (3, "Чип «Аврора» — выбранная персона (нажмите × для удаления)"),
        (4, "Чип «Аэлита» — вторая выбранная персона"),
    ]))
    story.append(Spacer(1, 8))
    story.append(tip_block(
        "Совет: При фильтрации по нескольким персонам в поиске используется логика И — "
        "будут найдены только фото, где присутствуют ВСЕ выбранные люди одновременно."
    ))
    story.append(PageBreak())

    # ===== 3. КАРТА =====
    story.append(Paragraph("3. Карта фотографий", STYLES["h1"]))

    # 3.1 Map overview
    story.append(Paragraph("3.1. Обзор карты", STYLES["h2"]))
    story.append(Paragraph(
        "Карта показывает фотографии, привязанные к GPS-координатам, "
        "в виде кластеров с миниатюрами. Число на кластере — количество фотографий в этой области.",
        STYLES["body"],
    ))
    story.append(Spacer(1, 4))
    story.append(img_flowable(img_paths["map_overview"]))
    story.extend(legend_items([
        (1, "Меню фильтров — форматы, персоны, даты"),
        (2, "Поиск — перейти на страницу текстового поиска по всей базе фотографий"),
        (3, "Карта (активная страница)"),
        (4, "Альбомы — перейти к альбомам"),
        (5, "Полный экран — развернуть карту на весь экран"),
        (6, "Кластер — нажмите для увеличения масштаба или открытия фото"),
        (7, "Переключатель слоёв — карта / спутник / тёмная тема"),
    ]))
    story.append(Spacer(1, 8))
    story.append(tip_block(
        "Совет: При наведении на кластер (или долгом нажатии) появится превью "
        "с миниатюрами первых 10 фотографий из этой области."
    ))
    story.append(Spacer(1, 10))

    # 3.2 Map filters
    story.append(Paragraph("3.2. Фильтры карты", STYLES["h2"]))
    story.append(Paragraph(
        "Панель фильтров карты содержит все инструменты для сужения выборки.",
        STYLES["body"],
    ))
    story.append(Spacer(1, 4))
    story.append(img_flowable(img_paths["map_filters"]))
    story.extend(legend_items([
        (1, "Секция «Форматы» — заголовок блока фильтров по типу файла"),
        (2, "Чекбокс формата — включите/выключите нужные форматы (применяется мгновенно)"),
        (3, "Фильтр по персоне — нажмите для выбора людей"),
        (4, "Дата «С» — начало периода"),
        (5, "Дата «По» — конец периода"),
        (6, "Статистика — количество фото с GPS из общего числа"),
        (7, "Закрыть панель фильтров"),
    ]))
    story.append(PageBreak())

    # 3.3 Person selector
    story.append(Paragraph("3.3. Выбор персоны", STYLES["h2"]))
    story.append(Paragraph(
        "Нажмите кнопку фильтра по персоне, чтобы открыть список людей.",
        STYLES["body"],
    ))
    story.append(Spacer(1, 4))
    story.append(img_flowable(img_paths["map_persons"], max_height=160 * mm))
    story.extend(legend_items([
        (1, "Кнопка с числом — количество выбранных персон"),
        (2, "Поле поиска — введите имя для быстрого поиска"),
        (3, "Галочка — персона выбрана (нажмите повторно для отмены)"),
        (4, "Миниатюра лица — фото-аватар распознанного человека"),
    ]))
    story.append(Spacer(1, 8))
    story.append(tip_block(
        "Совет: На карте фильтр по персонам использует логику ИЛИ — "
        "будут показаны фото, где есть ХОТЯ БЫ ОДИН из выбранных людей."
    ))
    story.append(Spacer(1, 10))

    # 3.4 Date filter
    story.append(Paragraph("3.4. Фильтр по дате", STYLES["h2"]))
    story.append(Paragraph(
        "Укажите период, чтобы видеть на карте только фотографии из этого диапазона дат.",
        STYLES["body"],
    ))
    story.append(Spacer(1, 4))
    story.append(img_flowable(img_paths["map_dates"], max_height=160 * mm))
    story.extend(legend_items([
        (1, "Бейдж «1» — одна персона выбрана в фильтре"),
        (2, "Дата начала — 14 февр. 2020 г."),
        (3, "Дата окончания — 28 февр. 2026 г."),
    ]))
    story.append(PageBreak())

    # ===== 4. ПРОСМОТР ФОТО =====
    story.append(Paragraph("4. Просмотр фотографий (Lightbox)", STYLES["h1"]))

    # 4.1 Lightbox
    story.append(Paragraph("4.1. Полноэкранный просмотр", STYLES["h2"]))
    story.append(Paragraph(
        "Нажмите на миниатюру фотографии, чтобы открыть её в полноэкранном режиме. "
        "Используйте стрелки или свайпы для навигации между фото.",
        STYLES["body"],
    ))
    story.append(Spacer(1, 4))
    story.append(img_flowable(img_paths["lightbox"]))
    story.extend(legend_items([
        (1, "Открыть на карте — показать место съёмки на карте (если есть GPS)"),
        (2, "Добавить в альбом — выбрать альбом для этого фото"),
        (3, "Показать лица — отобразить рамки распознанных лиц"),
        (4, "Закрыть — вернуться к результатам"),
        (5, "Предыдущее фото (или свайп вправо)"),
        (6, "Следующее фото (или свайп влево)"),
        (7, "Статусная строка — имя файла, процент совпадения, количество лиц (распознанные/всего)"),
    ]))
    story.append(Spacer(1, 10))

    # 4.2 Face detection
    story.append(Paragraph("4.2. Распознавание лиц", STYLES["h2"]))
    story.append(Paragraph(
        "Нажмите кнопку «Показать лица» в lightbox. "
        "На фотографии появятся рамки вокруг обнаруженных лиц. "
        "Нажмите на рамку лица, чтобы назначить или изменить персону.",
        STYLES["body"],
    ))
    story.append(Spacer(1, 4))
    story.append(img_flowable(img_paths["face_assign"]))
    story.extend(legend_items([
        (1, "Зелёная рамка — распознанное лицо (персона уже назначена)"),
        (2, "Красная рамка — выбранное лицо (открыт диалог назначения)"),
        (3, "Выпадающий список — выбор существующей персоны или ввод имени новой"),
        (4, "Уверенность — процент совпадения с известной персоной"),
        (5, "Сохранить — подтвердить назначение персоны"),
    ]))
    story.append(PageBreak())

    # ===== 5. АЛЬБОМЫ =====
    story.append(Paragraph("5. Альбомы", STYLES["h1"]))

    # 5.1 Albums page
    story.append(Paragraph("5.1. Список альбомов", STYLES["h2"]))
    story.append(Paragraph(
        "Страница альбомов показывает все созданные коллекции фотографий в виде карточек с обложками.",
        STYLES["body"],
    ))
    story.append(Spacer(1, 4))
    story.append(img_flowable(img_paths["albums_page"]))
    story.extend(legend_items([
        (1, "Поиск — перейти на страницу поиска"),
        (2, "Карта — перейти на карту фотографий"),
        (3, "Альбомы (текущая страница)"),
        (4, "Поиск альбомов — фильтрация по названию"),
        (5, "Создать новый альбом"),
        (6, "Карточка альбома — название, количество фото, обложка. Нажмите для открытия"),
    ]))
    story.append(Spacer(1, 10))

    # 5.2 Album picker
    story.append(Paragraph("5.2. Добавление фото в альбом", STYLES["h2"]))
    story.append(Paragraph(
        "При просмотре фото в lightbox нажмите кнопку «Добавить в альбом». "
        "Откроется модальное окно с выбором альбома.",
        STYLES["body"],
    ))
    story.append(Spacer(1, 4))
    story.append(img_flowable(img_paths["album_picker"]))
    story.extend(legend_items([
        (1, "Поиск — найти альбом по названию"),
        (2, "Создать новый альбом — если нужного альбома ещё нет"),
        (3, "Миниатюра обложки альбома"),
        (4, "Закрыть окно"),
    ]))
    story.append(Spacer(1, 8))
    story.append(tip_block(
        "Совет: Добавить несколько фото в альбом за раз можно из режима выделения — "
        "нажмите «Выбрать», отметьте нужные фотографии, затем используйте кнопку «В альбом»."
    ))

    # --- Build ---
    doc.build(story)
    print(f"\nPDF saved: {OUTPUT_PDF}")
    print(f"Annotated images: {ANNOTATED_DIR}")


if __name__ == "__main__":
    build_pdf()
