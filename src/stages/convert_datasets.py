import pandas as pd
from pathlib import Path
import argparse

from src.data import load_dataset
from src.utils.config import load_config
from src.utils.logging import get_logger

import json


def chatml_format(
    df: pd.DataFrame,
    instructions_col: str,
    output_col: str,
) -> list:
    chatml_data = []
    for _, example in df.iterrows():
        entry = {
            "messages": [
                {"role": "user", "content": example[instructions_col]},
                {"role": "assistant", "content": example[output_col]},
            ]
        }
        chatml_data.append(entry)
    return chatml_data


def convert_datasets(
    config_path: str,
    dataset: str,
) -> None:
    config = load_config(config_path)
    logger = get_logger("CONVERT_DATASET", config.base.log_level)

    output_dir = (
        Path(config.base.datasets_dir)
        / config.convert_datasets.convert_dir_path
        / dataset
    )
    if not output_dir.exists():
        logger.info(f"Directory {output_dir} does not exist. Creating it.")
        output_dir.mkdir(parents=True)
    else:
        raise FileExistsError(f"Directory {output_dir} already exists.")

    logger.info(f"Converting dataset {dataset} to ChatML format")
    dataset_path = (
        Path(config.base.datasets_dir)
        / config.format_datasets.raw_dir_path
        / config.data[dataset].dir_path
    )
    logger.info(f"Loading dataset {dataset} from {dataset_path}")
    dataset_df = load_dataset(
        dataset_path,
        config.data[dataset].format,
    )

    chatml_data = chatml_format(
        df=dataset_df,
        instructions_col=config.data[dataset].col_names.instruction,
        output_col=config.data[dataset].col_names.output,
    )

    chatml_path = output_dir / f"{dataset}.jsonl"
    logger.info(f"Saving ChatML formatted dataset to {chatml_path}")
    with open(chatml_path, "w") as f:
        for entry in chatml_data:
            json.dump(entry, f)
            f.write("\n")

    logger.info(f"Conversion complete")


if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args_parser.add_argument("--dataset", dest="dataset", required=True)
    args = args_parser.parse_args()
    convert_datasets(args.config, args.dataset)
