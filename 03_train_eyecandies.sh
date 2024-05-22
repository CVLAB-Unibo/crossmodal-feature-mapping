export CUDA_VISIBLE_DEVICES=0

dataset_path=datasets/eyecandies
checkpoint_savepath=models/checkpoints_CFM_eyecandies
epochs=50
batch_size=2

class_names=("CandyCane" "ChocolateCookie" "ChocolatePraline" "Confetto" "GummyBear" "HazelnutTruffle" "LicoriceSandwich" "Lollipop" "Marshmallow" "PeppermintCandy")

for class_name in "${class_names[@]}"
    do
        python cfm_training.py --class_name $class_name --model_type $model --epochs_no $epochs --batch_size $batch_size --dataset_path $dataset_path --checkpoint_savepath $checkpoint_savepath
    done