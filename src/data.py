import pandas as pd


def load_dataset(path: str, format: str) -> pd.DataFrame:
    """Load dataset from path

    Args:
        path (str): Path to dataset
        format (str): Format of dataset

    Returns:
        pd.DataFrame: Loaded dataset
    """
    if format == "csv":
        return pd.read_csv(path)
    elif format == "parquet":
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unknown dataset format {format}")
