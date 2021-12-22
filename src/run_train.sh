echo "Running file bash"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Running on linux"
    python train.py --root_path /media/data3/datasets/F-PHAB --model 2 --device cuda --epoch 50
elif [[ "$OSTYPE" == "msys" ]]; then
    echo "Running on window"
    python train.py --root_path ../hand_pose_action --model 2 --epoch 10
fi

python eval.py --weight weight --root_path ../hand_pose_action --subject Subject_2 --action put_salt --seq 1 --visualize True --model 2 

python hand_detect_util.py --weights runs/weights/best2.pt --source ../../hand_pose_action --img 416 --conf 0.4 --device 0

python eval.py -m 2 -d cuda -ds whd_obb_bound -in 1 -io 0 -s Subject_1 -a put_salt --seq 1 -v True -w train_result\03-12-2021-00-36-56