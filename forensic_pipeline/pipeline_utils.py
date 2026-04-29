from __future__ import annotations

from typing import Any, Iterable

import pandas as pd


MISSING_STRINGS = {"", "nan", "none", "nat", "null"}
TRUE_STRINGS = {"true", "1", "yes", "y", "t"}
FALSE_STRINGS = {"false", "0", "no", "n", "f", *MISSING_STRINGS}


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    return "" if text.strip().lower() in MISSING_STRINGS else text


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() in MISSING_STRINGS


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if is_missing(value):
        return False

    text = str(value).strip().lower()
    if text in TRUE_STRINGS:
        return True
    if text in FALSE_STRINGS:
        return False
    return False


def drop_blank_rows(df: pd.DataFrame, key_columns: Iterable[str] = ("media_id",)) -> pd.DataFrame:
    cleaned = df.copy()

    for col in key_columns:
        if col in cleaned.columns:
            mask = cleaned[col].map(lambda value: not is_missing(value))
            cleaned = cleaned.loc[mask].copy()

    if cleaned.empty:
        return cleaned.reset_index(drop=True)

    nonblank_any = cleaned.apply(
        lambda row: any(not is_missing(value) for value in row.values),
        axis=1,
    )
    return cleaned.loc[nonblank_any].reset_index(drop=True)
