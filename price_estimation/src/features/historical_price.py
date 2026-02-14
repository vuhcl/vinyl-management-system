"""
Historical price features: median past price, rolling median, time-decayed
price signal.
"""
import numpy as np
import pandas as pd


def add_time_decay_weights(
    df: pd.DataFrame,
    date_column: str = "sale_date",
    halflife_days: float = 365.0,
    weight_column: str = "time_weight",
) -> pd.DataFrame:
    """
    Add exponential time-decay weights so recent sales count more.
    weight = 0.5^((t_max - t) / halflife_days).
    """
    out = df.copy()
    if date_column not in out.columns:
        out[weight_column] = 1.0
        return out
    t = pd.to_datetime(out[date_column])
    t_max = t.max()
    days_ago = (t_max - t).dt.total_seconds() / (24 * 3600)
    out[weight_column] = np.power(0.5, days_ago / halflife_days)
    return out


def build_historical_price_features(
    df: pd.DataFrame,
    *,
    group_col: str = "item_id",
    price_col: str = "sale_price",
    date_col: str = "sale_date",
    rolling_window_days: float = 180.0,
    time_decay_halflife_days: float | None = 365.0,
    weight_col: str | None = "time_weight",
) -> pd.DataFrame:
    """
    Build per-item historical price aggregates:
    - median_price: median of all sale prices for the item
    - rolling_median_price: median over sales within rolling_window_days of each sale
    - time_decayed_mean: mean price weighted by time decay (recent = higher weight)

    Returns a DataFrame with one row per item_id (aggregated features). For
    point-in-time features we would compute per (item_id, sale_date); here we
    return item-level aggregates for training (one target per item).
    """
    if df.empty or price_col not in df.columns or group_col not in df.columns:
        return pd.DataFrame()

    out = df.copy()
    if date_col in out.columns:
        out[date_col] = pd.to_datetime(out[date_col])
    if time_decay_halflife_days is not None and date_col in out.columns:
        out = add_time_decay_weights(
            out,
            date_column=date_col,
            halflife_days=time_decay_halflife_days,
            weight_column=weight_col or "time_weight",
        )
    else:
        weight_col = None

    if weight_col and weight_col in out.columns:
        # Time-weighted mean: sum(price * weight) / sum(weight)
        out["_tw_sum"] = out[price_col] * out[weight_col]
        out["_w_sum"] = out[weight_col]
        tw = out.groupby(group_col, as_index=False).agg(
            _tw_sum=("_tw_sum", "sum"),
            _w_sum=("_w_sum", "sum"),
        )
        tw["time_decayed_mean"] = (
            tw["_tw_sum"] / tw["_w_sum"].replace(0, np.nan)
        )
        tw = tw[[group_col, "time_decayed_mean"]]
    else:
        tw = out[[group_col]].drop_duplicates()
        tw["time_decayed_mean"] = np.nan

    gr = out.groupby(group_col, as_index=False).agg(
        median_price=(price_col, "median"),
        mean_price=(price_col, "mean"),
        count_sales=(price_col, "count"),
    )
    gr = gr.merge(tw, on=group_col, how="left")

    # Rolling median: for each item, use only sales within window of latest sale
    if date_col in out.columns and rolling_window_days > 0:
        out = out.sort_values([group_col, date_col])
        out["_roll_end"] = out.groupby(group_col)[date_col].transform("max")
        out["_days_to_end"] = (
            (out["_roll_end"] - out[date_col]).dt.total_seconds() / (24 * 3600)
        )
        in_window = out[out["_days_to_end"] <= rolling_window_days]
        roll = in_window.groupby(group_col, as_index=False)[price_col].median()
        roll = roll.rename(columns={price_col: "rolling_median_price"})
        gr = gr.merge(roll, on=group_col, how="left")
    else:
        gr["rolling_median_price"] = gr["median_price"]

    return gr
