from mmengine.config import read_base
from opencompass.models import VLLMwithChatTemplate

with read_base():
    # Non-COT
    from opencompass.configs.datasets.VerififyBench.verifybench_hard_prompt_cv import verifier_datasets as verifybench_datasets

    # COT
    from opencompass.configs.datasets.VerififyBench.verifybench_hard_prompt_cv_cot import verifier_datasets as verifybench_datasets_cot
    from opencompass.configs.datasets.VerififyBench.verifybench_hard_prompt_verifybench import verifier_datasets as verifybench_datasets_cot_verifybench

    from opencompass.configs.models.qwen2_5.vllm_qwen2_5_3b_instruct import models as vllm_qwen2_5_3b_instruct_model
    # from opencompass.configs.models.qwen2_5.vllm_qwen2_5_7b_instruct import models as vllm_qwen2_5_7b_instruct_model
    # from opencompass.configs.models.qwen2_5.vllm_qwen2_5_14b_instruct import models as vllm_qwen2_5_14b_instruct_model
    # from opencompass.configs.models.qwen2_5.vllm_qwen2_5_32b_instruct import models as vllm_qwen2_5_32b_instruct_model
    # from opencompass.configs.models.qwen2_5.vllm_qwen2_5_72b_instruct import models as vllm_qwen2_5_72b_instruct_model
    # from opencompass.configs.models.openai.oss_120b import models as oss_120b_model
    # from opencompass.configs.models.openai.oss_20b import models as oss_20b_model

datasets = verifybench_datasets # verifybench_datasets_cot
models = sum([v for k, v in locals().items() if k.endswith('_model')], [])

# xverify_paths = [
#     "IAAR-Shanghai/xVerify-0.5B-I",
#     "IAAR-Shanghai/xVerify-8B-I",
#     "IAAR-Shanghai/xVerify-9B-C",
#     "IAAR-Shanghai/xVerify-3B-Ia"
# ]

# tencent_paths = [
#     "virtuoussy/Qwen2.5-7B-Instruct-RLVR"
# ]

# compassverifier_paths = [
#     "opencompass/CompassVerifier-3B",
#     "opencompass/CompassVerifier-7B",
#     "opencompass/CompassVerifier-32B"
# ]

# paths = compassverifier_paths # tencent_paths #  + xverify_paths 
# for path in paths:
#     models.append(
#         dict(
#             type=VLLMwithChatTemplate,
#             abbr=f'{path.replace("/", "-")}-vllm',
#             path=path,
#             model_kwargs=dict(tensor_parallel_size=1, max_model_len=32768),
#             # max_model_len=32768,
#             # max_seq_len=32768,
#             max_out_len=8192,
#             batch_size=16,
#             generation_kwargs=dict(temperature=0),
#             run_cfg=dict(num_gpus=2),
#         )
#     )


from opencompass.runners import LocalRunner
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=8),
    runner=dict(
        type=LocalRunner,
        max_num_workers=8,
        task=dict(type=OpenICLInferTask)
    ),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner, n=10),
    runner=dict(
        type=LocalRunner,
        max_num_workers=256,
        task=dict(type=OpenICLEvalTask)
    ),
)

work_dir = 'outputs/debug/verifybench'