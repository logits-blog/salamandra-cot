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
    progress_bar: tqdm = None,
) -> Dict[int, List[str]]:
    """Process lines in batches with LaTeX-aware splitting."""
    translated = defaultdict(list)

    for batch_start in range(0, len(lines), batch_size):
        batch = lines[batch_start : batch_start + batch_size]
        indices, raw_lines = zip(*batch)

        formatted = [
            prompt_template.format(
                src_lang=src_lang, tgt_lang=tgt_lang, stmt=line
            )
            for line in raw_lines
        ]

        inputs = tokenizer(
            formatted,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with (
            torch.no_grad(),
            torch.autocast(device_type="cuda", dtype=torch.float16),
        ):
            outputs = model.generate(
                **inputs,
                **model_args,
                return_dict_in_generate=True,
            )

        for i, (idx, output_seq) in enumerate(zip(indices, outputs.sequences)):
            prompt_len = inputs.input_ids[i].size(0)
            translated_text = tokenizer.decode(
                output_seq[prompt_len:],
                skip_special_tokens=True,
            ).strip()

            translated_line = split_at_outside_newline(translated_text)
            translated[idx].append(translated_line)

        if progress_bar:
            progress_bar.update(len(batch))

    return translated


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

    # Collect all lines with their original indices
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

    translated_data = defaultdict(lambda: {"instruction": [], "response": []})

    logger.info(f"Translating {len(instruction_lines)} instruction lines")
    with tqdm(total=len(instruction_lines), desc="Instruction Lines") as pbar:
        instr_results = process_text_lines(
            instruction_lines,
            model,
            tokenizer,
            config.dataset_translate.model.prompt_template,
            config.dataset_translate.model.model_args,
            dataset_conf.src_lang,
            dataset_conf.tgt_lang,
            device,
            batch_size,
            pbar,
        )
        for idx, lines in instr_results.items():
            translated_data[idx]["instruction"] = lines

    logger.info(f"Translating {len(response_lines)} response lines")
    with tqdm(total=len(response_lines), desc="Response Lines") as pbar:
        resp_results = process_text_lines(
            response_lines,
            model,
            tokenizer,
            config.dataset_translate.model.prompt_template,
            config.dataset_translate.model.model_args,
            dataset_conf.src_lang,
            dataset_conf.tgt_lang,
            device,
            batch_size,
            pbar,
        )
        for idx, lines in resp_results.items():
            translated_data[idx]["response"] = lines

    final_data = []
    for idx in range(len(df)):
        data = translated_data.get(idx, {"instruction": [], "response": []})
        final_data.append(
            {
                "original_index": idx,
                "original_instruction": df.iloc[idx].instruction,
                "original_response": df.iloc[idx].response,
                "translated_instruction": "\n".join(data["instruction"]),
                "translated_response": "\n".join(data["response"]),
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

        df = load_dataset(
            preprocessed_path,
            dataset_conf.format,
        ).reset_index(drop=True)

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
