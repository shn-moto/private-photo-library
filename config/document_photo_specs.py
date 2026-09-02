"""Document photo specifications, as data.

Adding a country/document = adding an entry here. No code changes.

Every number carries its provenance in `source`: where it came from, when it was
checked, and how much it is trusted. `confidence: "unverified_secondary"` means
the figure comes from commercial photo services rather than the official
regulation — the assistant must say so rather than presenting it as law.

Geometry is expressed as ratios of the output frame, calibrated against a real
accepted output (see `calibration`), not derived from anthropometric formulas —
an earlier attempt to compute head height from "eyes sit mid-head" was off by
~20% and produced out-of-spec crops.
"""

from typing import Dict, Optional

DOCUMENT_SPECS: Dict[str, dict] = {
    "pl_karta_pobytu_mos": {
        "country": "PL",
        "title": "Польша — карта побыту, загрузка в MOS",
        "description": (
            "Электронная подача через mos.cudzoziemcy.gov.pl. "
            "Система принимает JPG не меньше 684x883 px, до 2.5 МБ."
        ),
        # Output file
        "out_w": 684,
        "out_h": 883,
        "format": "JPEG",
        "quality": 92,
        "max_bytes": 2_560_000,
        # Physical equivalent (same frame, for printing on the plastic card)
        "print_mm": (35.0, 45.0),
        # Geometry as a fraction of frame height
        "head_ratio": 0.762,          # crown..chin / frame height
        "head_ratio_min": 0.70,
        "head_ratio_max": 0.80,
        "eye_from_bottom": 0.587,     # eye line, measured from the bottom edge
        "background_rgb": (255, 255, 255),
        # Human-readable rules the vision check evaluates
        "rules": [
            "нейтральное выражение лица, рот закрыт",
            "глаза открыты, взгляд прямо в объектив",
            "очки не допускаются, даже корректирующие",
            "головной убор не допускается, кроме религиозного",
            "волосы не закрывают овал лица, брови и глаза",
            "фон светлый, равномерный, без теней",
            "естественный оттенок кожи, без сильных фильтров и ретуши",
        ],
        "source": {
            "urls": [
                "https://passport-photo.online/pl/zdjecia-do-karty-pobytu-online",
                "https://mosfoto.com/pl",
                "https://fotodokarty.com.pl/pl/wymagania-zdjecia/",
            ],
            "checked_at": "2026-09-02",
            "confidence": "unverified_secondary",
            "note": (
                "Пиксельный минимум и лимит размера подтверждены несколькими "
                "источниками. Допуски по высоте головы официальным "
                "розпоряджением МСВіА не сверялись."
            ),
        },
        "calibration": {
            "note": (
                "Геометрия снята с реального результата mosfoto.com "
                "(684x883, принятый формат): голова 76.2% кадра = 34.3 мм, "
                "линия глаз 58.7% снизу, подбородок 5.7 мм от низа, фон 255."
            ),
            "measured_at": "2026-09-02",
        },
    },
}

# Same frame, printed rather than uploaded. Shares geometry, differs in output size.
DOCUMENT_SPECS["pl_karta_pobytu_print"] = {
    **DOCUMENT_SPECS["pl_karta_pobytu_mos"],
    "title": "Польша — карта побыту, печать 35x45 мм",
    "description": "Тот же кадр для печати на пластиковую карту, 35x45 мм при 600 DPI.",
    "out_w": 827,   # 35 mm @ 600 dpi
    "out_h": 1063,  # 45 mm @ 600 dpi
}


def get_spec(profile: str) -> Optional[dict]:
    return DOCUMENT_SPECS.get(profile)


def list_profiles() -> list:
    """Compact list for the assistant's tool schema and for the UI."""
    return [
        {
            "profile": key,
            "country": spec["country"],
            "title": spec["title"],
            "output": f"{spec['out_w']}x{spec['out_h']} {spec['format']}",
            "confidence": spec["source"]["confidence"],
        }
        for key, spec in DOCUMENT_SPECS.items()
    ]
