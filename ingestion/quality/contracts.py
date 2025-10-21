"""Data contracts for the ingestion layer.

We use the [Pandera](https://pandera.readthedocs.io/) library to define
schemas for our intermediate datasets.  Schemas act both as
documentation and as runtime validation.  Whenever the ingestion
pipeline produces a frame destined for the *silver* layer, it must
pass this schema; otherwise an exception will be raised.

See ``ingestion/transform/align.py`` for how the silver frame is
constructed.
"""

import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check


SilverSchema = DataFrameSchema(
    {
        "date": Column(pa.DateTime, coerce=True),
        "region_id": Column(str, nullable=False),
        "region_weight": Column(object, nullable=True),
        "temp_anom": Column(float, nullable=False, checks=Check(isfinite=True)),
        "precip_anom": Column(float, nullable=False, checks=Check(isfinite=True)),
        "ndvi": Column(float, nullable=True),
        "enso": Column(float, nullable=True),
        "prod_estimate": Column(float, nullable=True),
        "stocks": Column(float, nullable=True),
        "price_spot": Column(float, nullable=False),
        "price_front_fut": Column(float, nullable=False),
        "realized_vol_30d": Column(float, nullable=False, checks=Check(lambda x: x >= 0)),
    },
    strict=False,
    coerce=True,
    name="SilverLayer",
)

__all__ = ["SilverSchema"]