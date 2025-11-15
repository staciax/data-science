from contextlib import ExitStack

import matplotlib.font_manager as fm
from matplotlib import patheffects, rcParams as rcParams

__all__ = (
    'apply_luna_style',
    'luna',
)

# NOTE: luna is similar to xkcd with slight improvements
# NOTE: supports customizable font families
# TODO: find a handwriting-like font to use as default


def apply_luna_style(
    font_family: list[str] | None = None,
    scale: float = 0.2,
    length: float = 100,
    randomness: float = 1.5,
) -> None:
    rcParams.update(
        {
        'font.size': 10.0,
        'path.sketch': (scale, length, randomness),
        'path.effects': [patheffects.withStroke(linewidth=0.5, foreground='w')],
        'axes.linewidth': 1,
        'lines.linewidth': 2.0,
        'figure.facecolor': 'white',
        'grid.linewidth': 1.0,
        'axes.grid': False,
        'axes.unicode_minus': False,
        'axes.edgecolor': 'black',
        'xtick.major.size': 8,
        'xtick.major.width': 2,
        'ytick.major.size': 8,
        'ytick.major.width': 2,
    }
    )

    if font_family:
        cleaned_font_family = []
        for font in font_family:
            try:
                fm.findfont(font, fontext='ttf', fallback_to_default=False)
                cleaned_font_family.append(font)
            except ValueError:
                pass
        if cleaned_font_family:
            rcParams['font.family'] = list(set(cleaned_font_family))


def luna(
    *,
    font_family: list[str] | None = None,
    scale: float = 0.5,
    length: float = 100,
    randomness: float = 1.5,
) -> ExitStack:
    stack = ExitStack()
    stack.callback(rcParams._update_raw, rcParams.copy())  # type: ignore[arg-type]

    apply_luna_style(font_family=font_family, scale=scale, length=length, randomness=randomness)

    return stack
