from shutil import rmtree
import argparse
import os

from src.utils.config import load_config
from src.utils.logging import get_logger
from src.data import load_dataset

from tqdm import tqdm
import torch
import pandas as pd
from transformers import SeamlessM4Tv2Model, AutoProcessor
from transformers import AutoProcessor


def translate_batch(
    statements: list,
    model: SeamlessM4Tv2Model,
    processor: AutoProcessor,
    src_lang: str = "eng",
    tgt_lang: str = "spa",
    device: torch.device = torch.device("cuda:0"),
) -> list:
    """Translate a batch of statements

    Args:
        statements (list): List of statements to translate
        model (SeamlessM4Tv2Model): Translation model
        processor (AutoProcessor): Processor for the model
        src_lang (str): Source language code
        tgt_lang (str): Target language code
        device (torch.device): Device to run the model on

    Returns:
        list: List of translated statements
    """
    text_inputs = processor(
        text=statements,
        src_lang=src_lang,
        padding=True,
        return_tensors="pt",
    ).to(device)
    output_tokens = model.generate(
        **text_inputs,
        tgt_lang=tgt_lang,
        generate_speech=False,
    )
    translated_statements = [
        processor.decode(seq, skip_special_tokens=True)
        for seq in output_tokens.sequences
    ]
    # Clear cache
    torch.cuda.empty_cache()
    return [
        {"statement": statement, "translated_statement": translation}
        for statement, translation in zip(statements, translated_statements)
    ]


def join_partial_files(
    save_dir: str,
    naming: str,
) -> None:
    """Join partial files into a single file

    Args:
        save_dir (str): Directory where partial files are saved
        subset (str): Subset of the dataset
        naming (str): Naming convention for partial files
    """
    partial_files = [
        f
        for f in os.listdir(save_dir)
        if f.endswith(".csv") and f.startswith(f"{naming}_")
    ]
    partial_files.sort(key=lambda f: int(f.split("_")[-1].replace(".csv", "")))

    joined_file = os.path.join(save_dir, f"{naming}.csv")

    # Load each partial CSV and concatenate
    df_list = []
    for f in partial_files:
        df_list.append(pd.read_csv(os.path.join(save_dir, f)))
    df_joined = pd.concat(df_list, ignore_index=True)
    df_joined.to_csv(joined_file, index=False)

    for f in partial_files:
        os.remove(os.path.join(save_dir, f))


def translate_datasets(
    config_path: str,
    batch_size: int = 25,
    device="cuda:0",
) -> None:
    """Format datasets to a common standard

    Args:
        config_path (str): Path to configuration file
    """
    config = load_config(config_path)
    logger = get_logger("TRANSLATE_DATASETS", config.base.log_level)

    model_path = (
        f"{config.base.models_dir}/{config.dataset_translate.model.dir_path}"
    )
    logger.info(f"Loading translation model and processor from {model_path}")
    processor = AutoProcessor.from_pretrained(model_path)
    model = SeamlessM4Tv2Model.from_pretrained(model_path).to(device)

    for dataset, dataset_conf in config.data.train.items():
        dataset_path = f"{config.base.datasets_dir}/{config.format_datasets.format_dir_path}/{dataset}.parquet"
        save_dir = os.path.join(
            config.base.datasets_dir,
            config.dataset_translate.translate_dir_path,
            dataset,
        )

        if os.path.exists(save_dir):
            logger.info(f"Removing existing directory {save_dir}")
            rmtree(save_dir)
        os.makedirs(save_dir)

        logger.info(f"Loading dataset {dataset} from {dataset_path}")
        dataset_df = load_dataset(
            dataset_path,
            dataset_conf.format,
        )

        # Translate specified columns
        for col, _ in dataset_conf.col_names.items():
            if col not in dataset_df.columns:
                raise ValueError(f"Column {col} not found in dataset")

            statements = dataset_df[col].values.tolist()

            for idx in tqdm(
                range(0, len(statements), min(batch_size, len(statements)))
            ):
                chunk = statements[idx : idx + batch_size]
                save_file = os.path.join(
                    save_dir, f"{col}_translated_{idx}.csv"
                )

                translations = translate_batch(
                    chunk,
                    model,
                    processor,
                    dataset_conf.src_lang,
                    dataset_conf.tgt_lang,
                    device,
                )
                translations_df = pd.DataFrame(translations)
                translations_df.to_csv(save_file, index=False)
                logger.info(f"Saved generation results to {save_file}")

            logger.info(f"Joining partial files for {dataset}, {col}")
            join_partial_files(save_dir, f"{col}_translated")


if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args_parser.add_argument(
        "--batch-size", dest="batch_size", default=10, type=int
    )
    args = args_parser.parse_args()
    translate_datasets(args.config, args.batch_size)
