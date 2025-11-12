
"""
Market data ingestion from Yahoo Finance (robust via yfinance).

- Télécharge les prix journaliers des futures (Yahoo) puis agrège en hebdo (vendredi).
- Utilise 'Adj Close' si disponible, sinon 'Close'.
- Calcule la volatilité réalisée 30j annualisée à partir des log-returns.
- Fallback déterministe (synthetic) si Yahoo n'est pas disponible.
- Ajoute une colonne 'data_source' pour tracer la provenance ('yahoo' ou 'synthetic').

Sortie hebdo (W-FRI), colonnes:
    ['date', 'price_spot', 'price_front_fut', 'realized_vol_30d', 'data_source']
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class MarketClient:
    """Client for fetching commodity futures prices from Yahoo Finance via yfinance."""

    # Mapping générique -> ticker Yahoo
    _SYMBOL_MAP = {
        "wheat": "ZW=F",
        "corn": "ZC=F",
        "soybean": "ZS=F",
        "soybeans": "ZS=F",
        "cocoa": "CC=F",
        "coffee": "KC=F",
        "sugar": "SB=F",
    }

    def __init__(self, session: Optional[object] = None) -> None:
        # Gardé pour compat (pas utilisé par yfinance)
        self.session = session

    def _download_csv(self, symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
        """
        Télécharge OHLCV via yfinance de manière stable.

        - auto_adjust=False pour conserver 'Close' ET 'Adj Close'
          (le défaut récent de yfinance est True → on force False).
        - Normalise toujours en DataFrame et force l'index en UTC.
        """
        try:
            df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False)
        except Exception as exc:
            logger.warning("yfinance download failed for %s: %s", symbol, exc)
            return None

        if df is None or len(df) == 0:
            return None

        # yfinance peut renvoyer un Series: uniformiser en DataFrame
        if isinstance(df, pd.Series):
            df = df.to_frame(name="Close")

        # Index en UTC
        df.index = pd.to_datetime(df.index, utc=True)

        # Aplatir les colonnes si MultiIndex (e.g., ('Adj Close','ZS=F'))
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["|".join([str(x) for x in tup if str(x) != ""]) for tup in df.columns]

        # Parfois yfinance renvoie une unique colonne 2D (shape (n,1)); on squeeze
        for col in df.columns:
            col_vals = df[col].values
            if isinstance(col_vals, np.ndarray) and df[col].ndim == 2 and col_vals.shape[1] == 1:
                df[col] = pd.Series(col_vals.ravel(), index=df.index)

        return df  # colonnes possibles: Open, High, Low, Close, Adj Close, Volume

    def _pick_close_series(self, daily: pd.DataFrame) -> pd.Series:
        """
        Choisit de manière robuste une série de prix (Adj Close > Close),
        même si les colonnes ont été aplaties ou sont 2D.
        """
        cols = list(daily.columns)

        def find_col(names):
            for name in names:
                # exact match
                if name in daily.columns:
                    return daily[name]
                # version aplatie avec ticker: "Adj Close|ZS=F" / "Close|ZS=F"
                matches = [c for c in cols if c.startswith(name + "|")]
                if matches:
                    return daily[matches[0]]
            return None

        # priorité Adj Close puis Close
        s = find_col(["Adj Close", "Close"])
        if s is None:
            # dernier recours: première colonne
            s = daily.iloc[:, 0]

        # S'assurer que c'est bien une Series 1D
        if isinstance(s, pd.DataFrame):
            s = s.squeeze("columns")
        if hasattr(s, "values") and isinstance(s.values, np.ndarray) and getattr(s, "ndim", 1) == 2 and s.values.shape[1] == 1:
            s = pd.Series(s.values.ravel(), index=s.index, name=s.name)

        s = pd.to_numeric(s, errors="coerce")
        s.name = "Close"
        return s

    def fetch_prices(self, commodity: str, start: str, end: str) -> pd.DataFrame:
        """
        Renvoie les prix hebdo (vendredi) + vol 30j.
        Fallback reproductible si Yahoo indisponible.
        """
        key = commodity.lower()
        symbol = self._SYMBOL_MAP.get(key)
        if not symbol:
            logger.error("Unknown commodity '%s' in MarketClient", commodity)
            return pd.DataFrame(
                columns=["date", "price_spot", "price_front_fut", "realized_vol_30d", "data_source"]
            )

        # 1) Tente Yahoo/yfinance
        daily = self._download_csv(symbol, start, end)
        source = "yahoo"

        # 2) Fallback synthétique si besoin
        if daily is None or daily.empty:
            logger.info("Generating synthetic price path for %s (fallback)", commodity)
            source = "synthetic"
            dates = pd.date_range(start, end, freq="D", tz="UTC")
            rng = np.random.default_rng(seed=(hash(key) % 987654))
            dt = 1.0 / 252.0
            mu = 0.05
            sigma = 0.20

            # Ordres de grandeur plus réalistes par produit (indicatif)
            DEFAULTS = {
                "soybean": 12.0,    # USD/bu
                "corn": 5.0,        # USD/bu
                "wheat": 6.5,       # USD/bu
                "coffee": 180.0,    # attention unités contrat
                "cocoa": 3000.0,    # USD/tonne
                "sugar": 18.0,      # cts/lb approx
            }
            price0 = DEFAULTS.get(key, 100.0)

            prices = [price0]
            for _ in range(1, len(dates)):
                shock = rng.normal(loc=(mu - 0.5 * sigma**2) * dt, scale=sigma * np.sqrt(dt))
                prices.append(prices[-1] * np.exp(shock))
            daily = pd.DataFrame({"Close": prices}, index=dates)

        # 3) Choisir la série de prix la plus pertinente
        close_series = self._pick_close_series(daily)

        # 4) Log-returns & vol 30j annualisée (std rolling des log-returns * sqrt(252))
        log_ret = np.log(close_series).diff()
        vol_30 = log_ret.fillna(0).rolling(window=30, min_periods=1).std(ddof=0) * np.sqrt(252)
        vol_30.name = "vol_30"

        # 5) Construire un DF de travail avant resample (évite KeyError & erreurs 2D)
        work = pd.concat([close_series.rename("Close"), vol_30], axis=1)

        # 6) Agrégation hebdo sur VENDREDI
        weekly = work.resample("W-FRI").last()

        # 7) Colonnes finales
        weekly = weekly.rename(columns={"Close": "price_spot", "vol_30": "realized_vol_30d"})
        weekly["price_front_fut"] = weekly["price_spot"]  # approx front = spot (améliorable plus tard)
        weekly["data_source"] = source

        # >>> IMPORTANT : enlever le timezone (naive datetime) pour matcher météo/agri
        weekly.index = weekly.index.tz_convert(None)

        weekly.index.name = "date"
        weekly = weekly.reset_index()

        return weekly[["date", "price_spot", "price_front_fut", "realized_vol_30d", "data_source"]]


__all__ = ["MarketClient"]
