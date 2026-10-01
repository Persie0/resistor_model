BAND_COLORS = [
    "black", "brown", "red", "orange", "yellow", "green", "blue",
    "violet", "gray", "white", "gold", "silver",
]
COLOR_TO_INDEX = {name: i for i, name in enumerate(BAND_COLORS)}
INDEX_TO_COLOR = {i: name for name, i in COLOR_TO_INDEX.items()}
BACKGROUND_INDEX = len(BAND_COLORS)
DENSE_CLASSES = BAND_COLORS + ["body"]
