#!/bin/bash

### group2-4
# 探究不同image_prompt_ema_w大小的作用

as=("0.09" "0.07" "1.01" "1.03" "0.05" "1.05")

for image_prompt_ema_w in "${as[@]}"
do
    python ./tpt_ema.py /data/dataset/liuzichen/ \
        --test_sets A  --tpt --myclip  \
        --text_prompt_ema --text_prompt_ema_one_weight \
        --text_prompt_ema_one_weight_h=5000 \
        --text_prompt_ema_w=0.1 --image_prompts \
        --image_prompt_ema=4 --image_prompt_ema_h=5000 \
        --image_prompt_ema_w=${image_prompt_ema_w} \
        --info=A/This_CSTP-aEMA-h=5000-w=0.1-CSIP-r-aEMA-h=5000-w=${image_prompt_ema_w}-350- \
        --resize_flag=True --resize=350 \
        --resolution=224
done