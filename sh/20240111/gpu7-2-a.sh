#!/bin/bash

### group2-4
# 探究不同resize大小的作用

as=("340" "360" "370" "400")

for resize in "${as[@]}"
do
    python ./tpt_ema.py /data/dataset/liuzichen/ \
        --test_sets A  --tpt --myclip  \
        --text_prompt_ema --text_prompt_ema_one_weight \
        --text_prompt_ema_one_weight_h=5000 \
        --text_prompt_ema_w=0.1 --image_prompts \
        --image_prompt_ema=4 --image_prompt_ema_h=5000 \
        --image_prompt_ema_w=0.1 \
        --info=A/This_CSTP-aEMA-h=5000-w=0.1-CSIP-r-aEMA-h=5000-w=0.1-${resize}- \
        --resize_flag=True --resize=${resize} \
        --resolution=224
done