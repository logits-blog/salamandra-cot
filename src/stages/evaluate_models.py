from datetime import timedelta
import argparse

from src.utils.config import load_config
from src.utils.logging import get_logger

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.vllm.vllm_model import VLLMModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.utils.utils import EnvConfig
from lighteval.utils.imports import is_accelerate_available

if is_accelerate_available():
    from accelerate import Accelerator, InitProcessGroupKwargs

    accelerator = Accelerator(
        kwargs_handlers=[
            InitProcessGroupKwargs(timeout=timedelta(seconds=3000))
        ]
    )
else:
    accelerator = None


def evaluate_models(
    config_path: str,
) -> None:
    config = load_config(config_path)
    logger = get_logger("CONVERT_DATASET", config.base.log_level)

    evaluation_tracker = EvaluationTracker(
        output_dir=config.base.evaluations_dir,
        save_details=True,
        push_to_hub=True,
        hub_results_org=config.base.hf_org_id,
    )

    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.ACCELERATE,
        env_config=EnvConfig(cache_dir="tmp/"),
        # TODO: Remove the 2 parameters below once your configuration is tested
        override_batch_size=1,
        max_samples=10,
    )

    for model in config.evaluate_models.models:
        model_config = VLLMModelConfig(
            pretrained=config.models_dir[model],
            dtype="float16",
            use_chat_template=True,
        )

        pipeline = Pipeline(
            tasks=config.evaluate_models.tasks,
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
