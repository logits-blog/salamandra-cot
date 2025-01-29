import argparse
import os

from src.data import load_dataset
from src.utils.config import load_config
from src.utils.logging import get_logger


def format_datasets(
    config_path: str,
) -> None:
    """Format datasets to a common standard

    Args:
        config_path (str): Path to configuration file
    """
    config = load_config(config_path)
    logger = get_logger("DATA_PREPROCESS", config.base.log_level)

    for dataset, dataset_conf in config.data.items():
        dataset_path = f"{config.base.datasets_dir}/{config.format_datasets.raw_dir_path}/{dataset_conf.dir_path}"
        logger.info(f"Loading dataset {dataset} from {dataset_path}")
        dataset_df = load_dataset(
            dataset_path,
            dataset_conf.format,
        )

        logger.info(f"Formatting dataset {dataset}")
        dataset_df = dataset_df.rename(
            columns={
                dataset_conf.col_names.instruction: "instruction",
                dataset_conf.col_names.output: "output",
            }
        )[["instruction", "output"]]

        formatted_dir = f"{config.base.datasets_dir}/{config.format_datasets.format_dir_path}"
        formatted_path = f"{formatted_dir}/{dataset}.parquet"

        if not os.path.exists(formatted_dir):
            logger.info(
                f"Directory {formatted_dir} does not exist. Creating it."
            )
            os.makedirs(formatted_dir)

        logger.info(f"Saving formatted dataset to {formatted_path}")
        dataset_df.to_parquet(
            formatted_path,
            index=False,
        )


if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()
    format_datasets(args.config)
