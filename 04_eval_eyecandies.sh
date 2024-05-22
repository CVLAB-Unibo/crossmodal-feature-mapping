export CUDA_VISIBLE_DEVICES=1

dataset_path=datasets/eyecandies
checkpoint_savepath=models/checkpoints_CFM_eyecandies
epochs=50
batch_size=2

quantitative_folder=results/quantitatives_eyecandies

class_names=("CandyCane" "ChocolateCookie" "ChocolatePraline" "Confetto" "GummyBear" "HazelnutTruffle" "LicoriceSandwich" "Lollipop" "Marshmallow" "PeppermintCandy")

for class_name in "${class_names[@]}"
    do
        python cfm_inference.py --class_name $class_name --model_type $model --epochs_no $epochs --batch_size $batch_size --dataset_path $dataset_path --checkpoint_folder $checkpoint_savepath --quantitative_folder $quantitative_folder --produce_qualitatives
    done