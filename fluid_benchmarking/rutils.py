import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.vectors import DataFrame as RDataFrame


def pandas2r(
    df: pd.DataFrame,
) -> RDataFrame:
    with (ro.default_converter + pandas2ri.converter).context():
        df_r = ro.conversion.get_conversion().py2rpy(
            df
        )
    return df_r
