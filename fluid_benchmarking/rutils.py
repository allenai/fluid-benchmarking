from typing import Iterable

import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.vectors import (
    DataFrame as RDataFrame, 
    IntVector as RIntVector,
)


def df2r(
    df: pd.DataFrame,
) -> RDataFrame:
    with (ro.default_converter + pandas2ri.converter).context():
        df_r = ro.conversion.get_conversion().py2rpy(
            df.reset_index(drop=True)
        )
    return df_r


def vector2r(
    vector: Iterable[int],
) -> RIntVector:
    return ro.IntVector(vector)
