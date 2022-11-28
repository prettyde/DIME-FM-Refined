############## Configuration section begins ##################

# Model Config: [vitb32_CLIP, vitb16_CLIP, mae_vitb16, mocov3_vitb16, vit_base_patch16_224, vit_base_patch32_224, deit_base_patch16_224]
model_cfg=$MODEL_CFG

# Mode: [linear_probe, finetune, zeroshot]
mode=$MODE

# Use FP32 [default: True]
use_fp32=True

# Dataset: [caltech101]
dataset=$DATASET

# Model checkpoint
model_ckpt=$CKPT

# output directory
output_dir=$OUTPUT_DIR

############ Configurations for hyperparameter tuning begin ############
# set to True to disable the automatic hyperparameter tuning
# and set the learning rate and weight accordingly below
# This option is only effective for linear probe and finetuning.

disable_hyperparameter_tuning=False
learning_rate=0.1
l2_weight_decay=1e-6

############ Configurations for hyperparameter tuning end   ############

############ Configurations for linear_probe/finetune begin ############

# Random seed: [0,1,2]
random_seed=0

# Shots: {5, 20, 50} for few shot, and -1 for full-shot
num_shots=-1

# Whether to init the linear head with the text encoder
init_head_with_text_encoder=True

# whether to merge the encoder and the linear head
merge_encoder_and_proj=False

############ Configurations for linear_probe/finetune end   ############

############ Configurations for adding knowledge begin ############
# Please change the knowledge source accordingly.

use_wordnet_hierachy=False
use_wordnet_definition=False
use_wiktionary_definition=False
use_gpt3=False
use_gpt3_count=0

############ Configurations for adding knowledge end   ############

############## Configuration section ends ##################


# Launching the job......

cd vision_benchmark

if [[ "$mode" = "linear_probe" ]