echo "Running file bash"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Running on linux"
    python train.py --root_path /media/data3/datasets/F-PHAB --model 2 --device cuda --epoch 50
elif [[ "$OSTYPE" == "msys" ]]; then
    echo "Running on window"
    python train.py --root_path ../hand_pose_action --model 2 --epoch 10
fi