import json
from functools import lru_cache

from app.config import DISEASE_INFO_PATH


@lru_cache(maxsize=1)
def _load_info() -> dict:
    if not DISEASE_INFO_PATH.exists():
        return {}
    with open(DISEASE_INFO_PATH, "r") as f:
        return json.load(f)


def get_disease_info(class_label: str) -> dict:
    """Looks up symptoms/causes/prevention/treatment for a predicted class
    label. Falls back to a generic 'unknown' record if the label isn't in
    the knowledge base yet, so the API never 500s on a class we haven't
    documented."""
    info = _load_info().get(class_label)
    if info:
        return info

    plant, _, disease = class_label.partition("___")
    return {
        "plant": plant.replace("_", " ") or None,
        "disease": disease.replace("_", " ") or class_label,
        "is_healthy": "healthy" in class_label.lower(),
        "symptoms": None,
        "causes": None,
        "prevention": None,
        "treatment": "No entry yet in the disease knowledge base for this class -- add one to data/disease_info.json.",
    }
