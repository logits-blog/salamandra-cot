import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import load_dataset
from src.utils.config import load_config
from src.utils.logging import get_logger


def split_at_outside_newline(text: str) -> str:
    """
    Find the first newline character that is not inside a LaTeX block (\[...\] or \(...\))
    and split the text there. Returns the text up to (but not including) that newline.
    """
    in_latex = False
    i = 0
    n = len(text)
    while i < n:
        if text[i] == "\\" and i + 1 < n:
            next_char = text[i + 1]
            if next_char in ["[", "("]:
                in_latex = True
                i += 2  # Skip over opening delimiter
                continue
            elif next_char in ["]", ")"]:
                in_latex = False
                i += 2  # Skip over closing delimiter
                continue
        elif text[i] == "\n" and not in_latex:
            return text[:i]  # Split here
        i += 1
    return text


def process_text_lines(
    lines: List[Tuple[int, str]],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_template: str,
    model_args: dict,
    src_lang: str,
    tgt_lang: str,
    device: torch.device,
    batch_size: int,
    checkpoint_dir: Path,
    checkpoint_step: int = 100,
    progress_bar: tqdm = None,
) -> None:
    """
    Process lines in batches with LaTeX-aware splitting. Saves checkpoints every `checkpoint_step` batches
    (if > 0) to checkpoint_dir. Each checkpoint contains data from batches since the last checkpoint.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    accumulator = []
    checkpoint_count = 0

    for batch_idx, batch_start in enumerate(
        range(0, len(lines), batch_size), start=1
    ):
        batch = lines[batch_start : batch_start + batch_size]
        indices, raw_lines = zip(*batch)

        formatted = [
            prompt_template.format(
                src_lang=src_lang, tgt_lang=tgt_lang, stmt=line
            )
            for line in raw_lines
        ]

        inputs = tokenizer(formatted, padding=True, return_tensors="pt").to(
            device
        )

        with (
            torch.no_grad(),
            torch.autocast(device_type="cuda", dtype=torch.float16),
        ):
            outputs = model.generate(
                **inputs,
                **model_args,
                return_dict_in_generate=True,
            )

        batch_translations = []
        for i, (idx, output_seq) in enumerate(zip(indices, outputs.sequences)):
            prompt_len = inputs.input_ids[i].size(0)
            translated_text = tokenizer.decode(
                output_seq[prompt_len:],
                skip_special_tokens=True,
            ).strip()

            translated_line = split_at_outside_newline(translated_text)
            batch_translations.append(
                {"original_index": idx, "translated_line": translated_line}
            )

        accumulator.extend(batch_translations)

        if progress_bar:
            progress_bar.update(len(batch))

        if checkpoint_step > 0 and (batch_idx % checkpoint_step == 0):
            checkpoint_count += 1
            checkpoint_path = (
                checkpoint_dir / f"checkpoint_{checkpoint_count:04d}.parquet"
            )
            pd.DataFrame(accumulator).to_parquet(checkpoint_path)
            accumulator = []
            if progress_bar:
                progress_bar.write(
                    f"Saved checkpoint {checkpoint_count} to {checkpoint_path}"
                )

    if accumulator:
        checkpoint_count += 1
        checkpoint_path = (
            checkpoint_dir / f"checkpoint_{checkpoint_count:04d}.parquet"
        )
        pd.DataFrame(accumulator).to_parquet(checkpoint_path)
        if progress_bar:
            progress_bar.write(
                f"Saved final checkpoint {checkpoint_count} to {checkpoint_path}"
            )


def merge_checkpoints(checkpoint_dir: Path) -> Dict[int, List[str]]:
    """Merge all checkpoint files in directory into a single dict grouped by original_index"""
    checkpoint_files = sorted(checkpoint_dir.glob("checkpoint_*.parquet"))
    merged = defaultdict(list)
    for ckpt_file in checkpoint_files:
        df = pd.read_parquet(ckpt_file)
        for _, row in df.iterrows():
            merged[row["original_index"]].append(row["translated_line"])
    return merged


def translate_dataset(
    df: pd.DataFrame,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: dict,
    dataset_conf: dict,
    device: torch.device,
    save_dir: Path,
    batch_size: int,
    logger: any,
) -> None:
    instruction_lines, response_lines = [], []
    for idx in range(len(df)):
        instruction = df.iloc[idx].instruction
        response = df.iloc[idx].response

        instruction_lines.extend(
            (idx, line.strip()) for line in instruction.split("\n")
        )
        response_lines.extend(
            (idx, line.strip()) for line in response.split("\n")
        )

    instruction_ckpt_dir = save_dir / "checkpoints_instruction"
    response_ckpt_dir = save_dir / "checkpoints_response"
    checkpoint_step = config.dataset_translate.checkpoint_step

    logger.info(f"Translating {len(instruction_lines)} instruction lines")
    with tqdm(total=len(instruction_lines), desc="Instruction Lines") as pbar:
        process_text_lines(
            lines=instruction_lines,
            model=model,
            tokenizer=tokenizer,
            prompt_template=config.dataset_translate.model.prompt_template,
            model_args=config.dataset_translate.model.model_args,
            src_lang=dataset_conf.src_lang,
            tgt_lang=dataset_conf.tgt_lang,
            device=device,
            batch_size=batch_size,
            checkpoint_dir=instruction_ckpt_dir,
            checkpoint_step=checkpoint_step,
            progress_bar=pbar,
        )

    logger.info(f"Translating {len(response_lines)} response lines")
    with tqdm(total=len(response_lines), desc="Response Lines") as pbar:
        process_text_lines(
            lines=response_lines,
            model=model,
            tokenizer=tokenizer,
            prompt_template=config.dataset_translate.model.prompt_template,
            model_args=config.dataset_translate.model.model_args,
            src_lang=dataset_conf.src_lang,
            tgt_lang=dataset_conf.tgt_lang,
            device=device,
            batch_size=batch_size,
            checkpoint_dir=response_ckpt_dir,
            checkpoint_step=checkpoint_step,
            progress_bar=pbar,
        )

    # Merge checkpoints and build final dataset
    logger.info("Merging instruction checkpoints")
    instr_results = merge_checkpoints(instruction_ckpt_dir)
    logger.info("Merging response checkpoints")
    resp_results = merge_checkpoints(response_ckpt_dir)

    # Build final translated dataset
    final_data = []
    for idx in range(len(df)):
        final_data.append(
            {
                "original_index": idx,
                "original_instruction": df.iloc[idx].instruction,
                "original_response": df.iloc[idx].response,
                "translated_instruction": "\n".join(instr_results.get(idx, [])),
                "translated_response": "\n".join(resp_results.get(idx, [])),
            }
        )

    final_path = save_dir / f"{dataset_conf.name}_translated.parquet"
    pd.DataFrame(final_data).to_parquet(final_path)
    logger.info(f"Saved translated dataset to {final_path}")


def translate_datasets(config_path: str, batch_size: int) -> None:
    """Main entry point with configuration handling."""
    config = load_config(config_path)
    logger = get_logger("TRANSLATE_DATASETS", config.base.log_level)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_path = (
        Path(config.base.models_dir) / config.dataset_translate.model.dir_path
    )
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    for dataset_name, dataset_conf in config.data.train.items():
        save_dir = (
            Path(config.base.datasets_dir)
            / config.dataset_translate.translate_dir_path
            / dataset_name
        )
        save_dir.mkdir(parents=True, exist_ok=True)

        preprocessed_path = (
            Path(config.base.datasets_dir)
            / config.preprocess_datasets.preprocess_dir_path
            / dataset_name
            / f"{dataset_name}.parquet"
        )
        if not preprocessed_path.exists():
            raise FileNotFoundError(
                f"Preprocessed dataset missing at {preprocessed_path}"
            )

        df = load_dataset(preprocessed_path, dataset_conf.format).reset_index(
            drop=True
        )

        translate_dataset(
            df=df,
            model=model,
            tokenizer=tokenizer,
            config=config,
            dataset_conf=dataset_conf,
            device=device,
            save_dir=save_dir,
            batch_size=batch_size,
            logger=logger,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=10,
    )
    args = parser.parse_args()
    translate_datasets(args.config, args.batch_size)
