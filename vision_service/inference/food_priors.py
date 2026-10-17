"""
Food class priors: typical height and volume for common foods.

Used to correct Depth Pro's per-photo scale error. On the 513-photo
SimpleFood45 sweep, Depth Pro reported food heights 5-40mm where real
foods are 20-70mm. This table is the correction target.

Values are typical, not exact. A bagel is 35-45mm tall; we use 40mm.
The table is expected to grow as new food classes are added.

Source: mixed, from USDA portion data and food-safety published sizes.
"""

FOOD_PRIORS = {
    "bagel":         {"height_mm": 40, "volume_cm3": 300},
    "croissant":     {"height_mm": 50, "volume_cm3": 200},
    "apple":         {"height_mm": 70, "volume_cm3": 250},
    "banana":        {"height_mm": 40, "volume_cm3": 180},
    "orange":        {"height_mm": 75, "volume_cm3": 200},
    "sandwich":      {"height_mm": 50, "volume_cm3": 350},
    "slice of bread": {"height_mm": 15, "volume_cm3": 80},
    "muffin":        {"height_mm": 60, "volume_cm3": 180},
    "donut":         {"height_mm": 40, "volume_cm3": 120},
    "cookie":        {"height_mm": 15, "volume_cm3": 30},
    "pizza slice":   {"height_mm": 15, "volume_cm3": 120},
    "burger":        {"height_mm": 60, "volume_cm3": 400},
    "hot dog":       {"height_mm": 45, "volume_cm3": 200},
    "salad":         {"height_mm": 60, "volume_cm3": 300},
    "rice":          {"height_mm": 30, "volume_cm3": 200},
    "pasta":         {"height_mm": 40, "volume_cm3": 250},
    "chicken breast": {"height_mm": 35, "volume_cm3": 200},
    "steak":         {"height_mm": 35, "volume_cm3": 250},
    "egg":           {"height_mm": 60, "volume_cm3": 50},
    "cheese slice":  {"height_mm": 10, "volume_cm3": 30},
}

DEFAULT_PRIOR = {"height_mm": 40, "volume_cm3": 250}


def get_prior(label: str) -> dict:
    """Return the prior for a food class, or the default."""
    if not label:
        return DEFAULT_PRIOR
    return FOOD_PRIORS.get(label.lower().strip(), DEFAULT_PRIOR)
