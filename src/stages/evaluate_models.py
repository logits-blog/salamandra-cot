import os
import argparse

from src.utils.config import load_config
from src.utils.logging import get_logger

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.transformers.transformers_model import (
    TransformersModelConfig,
)
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.utils.utils import EnvConfig

os.environ["CUDA_VISIBLE_DEVICES"] = "cuda:0"


def evaluate_models(
    config_path: str,
) -> None:
    config = load_config(config_path)
    logger = get_logger("CONVERT_DATASET", config.base.log_level)

    evaluation_tracker = EvaluationTracker(
        output_dir=config.base.evaluations_dir,
        save_details=True,
        push_to_hub=False,
    )

    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.ACCELERATE,
        env_config=EnvConfig(cache_dir="tmp/"),
        # TODO: Remove the 2 parameters below once your configuration is tested
        override_batch_size=1,
        max_samples=10,
    )

    for model, model_conf in config.evaluate_models.models.items():
        logger.info(f"Evaluating model: {model}")

        model_config = {
            "pretrained": f"{config.base.models_dir}/{model_conf.dir_path}",
            "dtype": "bfloat16",
            "use_chat_template": True,
        }

        model_config = TransformersModelConfig(
            **model_config,
        )

        few_shot = config.evaluate_models.few_shot
        truncate_few_shots = config.evaluate_models.truncate_few_shots
        tasks = ",".join(
            [
                f"{t}|{few_shot}|{truncate_few_shots}"
                for t in config.evaluate_models.tasks
            ]
        )

        logger.info("Evaluation tasks:\n" + "\n".join(config.evaluate_models.tasks))
        logger.info(f"Few shot: {few_shot}, Truncate few shots: {truncate_few_shots}")

        pipeline = Pipeline(
            tasks=tasks,
            pipeline_parameters=pipeline_params,
            evaluation_tracker=evaluation_tracker,
            model_config=model_config,
        )

        pipeline.evaluate()
        pipeline.save_and_push_results()
        pipeline.show_results()

        del model_config
        del pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    evaluate_models(args.config)
