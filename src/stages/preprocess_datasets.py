import argparse
import random
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

from src.data import load_dataset
from src.utils.config import load_config
from src.utils.logging import get_logger


def filter_dataset(config_path: str) -> None:
    """Optimized filtering with adaptive sampling."""
    config = load_config(config_path)
    logger = get_logger("FILTER_DATASET", config.base.log_level)

    tokenizer = AutoTokenizer.from_pretrained(
        Path(config.base.models_dir) / config.dataset_translate.model.dir_path
    )

    for dataset_name, dataset_conf in config.data.train.items():
        input_path = (
            Path(config.base.datasets_dir)
            / config.format_datasets.format_dir_path
            / f"{dataset_name}.parquet"
        )
        output_dir = (
            Path(config.base.datasets_dir)
            / config.preprocess_datasets.preprocess_dir_path
            / dataset_name
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{dataset_name}.parquet"

        if output_path.exists():
            logger.info(f"Skipping existing: {dataset_name}")
            continue

        logger.info(f"Processing: {dataset_name}")
        df = load_dataset(input_path, dataset_conf.format)

        max_samples = dataset_conf.max_samples
        max_length = dataset_conf.max_length
        prompt_template = config.dataset_translate.model.prompt_template
        src_lang = dataset_conf.src_lang
        tgt_lang = dataset_conf.tgt_lang

        char_per_token = 4  # Conservative estimate for non-whitespace languages
        buffer_factor = 1.5 if max_samples else 1
        target_samples = (
            int(max_samples * buffer_factor) if max_samples else None
        )

        if target_samples:
            candidate_indices = random.sample(
                range(len(df)), min(target_samples, len(df))
            )
        else:
            candidate_indices = range(len(df))

        valid_indices = []
        template_len = len(
            prompt_template.format(
                src_lang=src_lang, tgt_lang=tgt_lang, stmt=""
            )
        )

        for idx in tqdm(candidate_indices, desc="Processing candidates"):
            row = df.iloc[idx]
            try:
                instr_len = len(row.instruction) + template_len
                resp_len = len(row.response) + template_len
                if (
                    instr_len > max_length * char_per_token
                    or resp_len > max_length * char_per_token
                ):
                    continue

                formatted_instr = prompt_template.format(
                    src_lang=src_lang, tgt_lang=tgt_lang, stmt=row.instruction
                )
                formatted_resp = prompt_template.format(
                    src_lang=src_lang, tgt_lang=tgt_lang, stmt=row.response
                )

                instr_tokens = tokenizer.encode(
                    formatted_instr, max_length=max_length, truncation=True
                )
                resp_tokens = tokenizer.encode(
                    formatted_resp, max_length=max_length, truncation=True
                )

                if (
                    len(instr_tokens) < max_length
                    and len(resp_tokens) < max_length
                ):
                    valid_indices.append(idx)

            except Exception as e:
                logger.error(f"Error {idx}: {str(e)}")

        # Final sampling if needed
        if max_samples:
            valid_indices = valid_indices[:max_samples]

        pd.DataFrame(df.iloc[valid_indices]).to_parquet(output_path)
        logger.info(f"Saved filtered dataset to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    filter_dataset(args.config)
