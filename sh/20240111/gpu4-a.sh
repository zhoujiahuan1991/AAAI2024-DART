#!/bin/bash

### group2-4
# 探究不同h大小的作用

as=("4950" "4900" "4850" "4800" "4750" "4700")
# as=("0.09" "0.07" "1.01" "1.03" "0.05" "1.05")
image_prompt_ema_w=0.1
text_prompt_ema_w=0.1

for h in "${as[@]}"
do
    python ./tpt_ema.py /data/dataset/liuzichen/ \
        --test_sets A  --tpt --myclip  \
        --text_prompt_ema --text_prompt_ema_one_weight \
        --text_prompt_ema_one_weight_h=${h} \
        --text_prompt_ema_w=${text_prompt_ema_w} --image_prompts \
        --image_prompt_ema=4 --image_prompt_ema_h=${h} \
        --image_prompt_ema_w=${image_prompt_ema_w} \
        --info=A/This_CSTP-aEMA-h=${h}-w=${text_prompt_ema_w}-CSIP-r-aEMA-h=${h}-w=${image_prompt_ema_w}-350- \
        --resize_flag=True --resize=350 \
        --resolution=224
done