export CUDA_VISIBLE_DEVICES=1

epochs=250
batch_size=4

class_names=("bagel" "cable_gland" "carrot" "cookie" "dowel" "foam" "peach" "potato" "rope" "tire")

for class_name in "${class_names[@]}"
    do
        python cfm_inference.py --class_name $class_name --epochs_no $epochs --batch_size $batch_size
    done