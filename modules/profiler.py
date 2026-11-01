
import pandas as pd
from ydata_profiling import ProfileReport


def generate_profile(df: pd.DataFrame) -> ProfileReport:
    """
    Generate a detailed statistical profile of a dataset.
    """

    if df is None or df.empty:
        raise ValueError("The dataset is empty.")

    profile = ProfileReport(
        df,
        title="DataSentinel - Dataset Quality Report",
        explorative=True,
        minimal=False,
        progress_bar=False
    )

    return profile
