#bin/bash
huggingface-cli download Magpie-Align/Magpie-Reasoning-V2-250K-CoT-Llama3 --local-dir ../datasets/magpie_v2_llama3 --repo-type dataset
huggingface-cli download facebook/seamless-m4t-v2-large --local-dir ../models/seamless_m4t_v2_large  