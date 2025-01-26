import argparse
import glob
import os
import re

from src.utils.config import load_config
from src.utils.logging import get_logger
from src.data import load_dataset

from tqdm import tqdm
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def translate_batch(
    statements: list,
    model: any,
    tokenizer: any,
    max_length: int,
    prompt_template: str,
    model_args: dict,
    src_lang: str = "en",
    tgt_lang: str = "es",
    device: torch.device = torch.device("cuda:0"),
) -> list:
    """Translate a batch of statements, skipping those longer than the specified max_length.

    Args:
        statements (list): List of statements to translate
        model (AutoModel): Translation model
        tokenizer (AutoProcessor): Tokenizer for the model
        max_length (int): Maximum allowed sequence length for the model
        prompt_template (str): Template for formatting input text. Should avoid including source/target language codes if tokenizer adds them.
        src_lang (str): Source language code
        tgt_lang (str): Target language code
        device (torch.device): Device to run the model on

    Returns:
        list: List of dictionaries with original and translated statements, in original order with invalid entries skipped.
    """
    statements_formatted = [
        prompt_template.format(src_lang=src_lang, tgt_lang=tgt_lang, stmt=stmt)
        for stmt in statements
    ]

    batch_encoding = tokenizer(
        text=statements_formatted,
        padding=False,
        truncation=False,
        return_tensors=None,
        add_special_tokens=True,
    )
    input_ids_list = batch_encoding["input_ids"]

    # Filter valid indices based on token lengths
    valid_indices = [
        i for i, ids in enumerate(input_ids_list) if len(ids) <= max_length
    ]
    valid_statements = [statements_formatted[i] for i in valid_indices]
    translated_valid = []

    if valid_statements:
        text_inputs = tokenizer(
            text=valid_statements,
            padding=True,
            truncation=False,
            return_tensors="pt",
            add_special_tokens=True,
        ).to(device)

        with (
            torch.no_grad(),
            torch.autocast(device_type="cuda", dtype=torch.float16),
        ):
            output_tokens = model.generate(
                **text_inputs,
                max_length=max_length,
                return_dict_in_generate=True,
                return_legacy_cache=True,
                **model_args,
            )

        translated_valid = tokenizer.batch_decode(
            output_tokens.sequences, skip_special_tokens=True
        )

        del text_inputs, output_tokens
        torch.cuda.empty_cache()

    translated_statements = []
    for idx, trans in zip(valid_indices, translated_valid):
        translated_statements.append(
            {
                "statement": statements[idx],
                "translated_statement": trans,
            }
        )

    return translated_statements


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
    batch_size: int,
    subsets: list = None,
    device: str = "cuda:0",
) -> None:
    """
    Translate dataset by:
      - Processing statements in GPU sub-batches of size `batch_size`.
      - Checkpointing every `checkpoint_size` statements.
      - Resuming from existing partial files if present.
      - Combining all partial files at the end.

    Args:
        config_path (str): Path to config file.
        batch_size (int): Number of statements per GPU sub-batch.
        subsets (list): Optional columns to process. If None, process all.
        device (str): e.g. 'cuda:0'.
    """
    config = load_config(config_path)
    logger = get_logger("TRANSLATE_DATASETS", config.base.log_level)

    checkpoint_size = config.dataset_translate.checkpoint_step

    model_path = os.path.join(
        config.base.models_dir,
        config.dataset_translate.model.dir_path,
    )
    logger.info(f"Loading translation model from {model_path}")

    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, padding_side="left"
    )  # Ensure tokenizer is set to left padding (decoder-only model)
    model.generation_config.pad_token_id = (
        tokenizer.pad_token_id
    )  # Ensure padding token is set
    model.eval()  # Set model to evaluation mode

    for dataset, dataset_conf in config.data.train.items():
        dataset_path = os.path.join(
            config.base.datasets_dir,
            config.format_datasets.format_dir_path,
            f"{dataset}.parquet",
        )
        save_dir = os.path.join(
            config.base.datasets_dir,
            config.dataset_translate.translate_dir_path,
            dataset,
        )
        os.makedirs(save_dir, exist_ok=True)

        logger.info(f"Loading dataset '{dataset}' from {dataset_path}")
        dataset_df = load_dataset(dataset_path, dataset_conf.format)

        for col, _ in dataset_conf.col_names.items():
            if subsets and col not in subsets:
                logger.info(f"Skipping {col} (not in subsets)")
                continue

            if col not in dataset_df.columns:
                raise ValueError(f"Column {col} not found in dataset")

            statements = dataset_df[col].tolist()
            total_statements = len(statements)
            logger.info(
                f"Column '{col}' has {total_statements} statements. "
                f"Processing sub-batches of {batch_size}, "
                f"checkpoint every {checkpoint_size} statements."
            )

            # Find partial files for resume
            # col_partial_{startIdx}_{endIdx}.csv
            pattern = re.compile(rf"{col}_partial_(\d+)_(\d+)\.csv$")
            existing_ranges = []
            for path_ in glob.glob(
                os.path.join(save_dir, f"{col}_partial_*.csv")
            ):
                fname = os.path.basename(path_)
                match = pattern.match(fname)
                if match:
                    s_idx = int(match.group(1))
                    e_idx = int(match.group(2))
                    existing_ranges.append((s_idx, e_idx))

            largest_end_idx = -1
            if existing_ranges:
                existing_ranges.sort(key=lambda x: x[0])
                if existing_ranges[0][0] != 0:
                    raise ValueError(
                        f"Partial files for {col} do not start at index 0. "
                        "Delete partial files or ensure they start at 0."
                    )
                # Check for continuity between partials
                for i in range(1, len(existing_ranges)):
                    prev_end = existing_ranges[i - 1][1]
                    curr_start = existing_ranges[i][0]
                    if curr_start != prev_end + 1:
                        raise ValueError(
                            f"Gap detected in partial files for {col} between "
                            f"{prev_end} and {curr_start}. Delete or fix partial files."
                        )
                largest_end_idx = existing_ranges[-1][1]
            else:
                largest_end_idx = -1

            current_idx = largest_end_idx + 1
            if current_idx < 0:
                current_idx = 0

            logger.info(
                f"Resuming translation for '{col}' starting from statement index {current_idx}."
            )

            aggregator = []
            agg_start_idx = current_idx
            statements_since_checkpoint = 0
            remaining = total_statements - current_idx
            pbar = tqdm(
                total=remaining, desc=f"Translating {col}", unit="statements"
            )

            # Translate loop in sub-batches of `batch_size`
            while current_idx < total_statements:
                sub_batch_end = min(current_idx + batch_size, total_statements)
                sub_batch = statements[current_idx:sub_batch_end]
                sub_translations = translate_batch(
                    sub_batch,
                    model,
                    tokenizer,
                    config.dataset_translate.model.max_length,
                    config.dataset_translate.model.prompt_template,
                    config.dataset_translate.model.model_args,
                    dataset_conf.src_lang,
                    dataset_conf.tgt_lang,
                    device,
                )
                aggregator.extend(sub_translations)
                statements_since_checkpoint += len(sub_batch)
                pbar.update(len(sub_batch))
                current_idx = sub_batch_end  # move to next sub-batch

                # Checkpointing
                if statements_since_checkpoint >= checkpoint_size:
                    partial_start = agg_start_idx
                    partial_end = (
                        agg_start_idx + statements_since_checkpoint - 1
                    )

                    partial_path = os.path.join(
                        save_dir,
                        f"{col}_partial_{partial_start}_{partial_end}.csv",
                    )
                    pd.DataFrame(aggregator).to_csv(partial_path, index=False)
                    logger.info(
                        f"Checkpoint: wrote {statements_since_checkpoint} statements "
                        f"[{partial_start}..{partial_end}] to {partial_path}"
                    )

                    # Reset aggregator
                    aggregator.clear()
                    agg_start_idx = partial_end + 1
                    statements_since_checkpoint = 0

            # Close progress bar
            pbar.close()

            # Aggregator has leftover statements (< checkpoint_size)
            if aggregator:
                partial_start = agg_start_idx
                partial_end = partial_start + len(aggregator) - 1
                partial_path = os.path.join(
                    save_dir, f"{col}_partial_{partial_start}_{partial_end}.csv"
                )
                pd.DataFrame(aggregator).to_csv(partial_path, index=False)
                logger.info(
                    f"Final partial: wrote leftover {len(aggregator)} statements "
                    f"[{partial_start}..{partial_end}] to {partial_path}"
                )
                aggregator.clear()

            # Combine all partial files
            partial_files = sorted(
                glob.glob(os.path.join(save_dir, f"{col}_partial_*.csv")),
                key=lambda x: (
                    int(re.findall(r"_partial_(\d+)_", x)[0]),
                    int(re.findall(r"_(\d+)\.csv", x)[0]),
                ),
            )

            if partial_files:
                combined_df = pd.concat(
                    (pd.read_csv(f) for f in partial_files), ignore_index=True
                )
                final_path = os.path.join(save_dir, f"{col}_translated.csv")
                combined_df.to_csv(final_path, index=False)

                # Verify completeness
                if combined_df.shape[0] == total_statements:
                    logger.info(
                        f"Combined partial files into '{final_path}' with {combined_df.shape[0]} rows. "
                        "Removing partial files."
                    )
                    for f in partial_files:
                        os.remove(f)
                else:
                    logger.error(
                        f"Combined file for '{col}' has {combined_df.shape[0]} rows, "
                        f"but expected {total_statements}. Keeping partial files."
                    )


if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args_parser.add_argument(
        "--batch-size", dest="batch_size", default=10, type=int
    )
    args_parser.add_argument(
        "--subsets", dest="subsets", nargs="+", default=None, type=str
    )
    args = args_parser.parse_args()
    translate_datasets(args.config, args.batch_size, args.subsets)
