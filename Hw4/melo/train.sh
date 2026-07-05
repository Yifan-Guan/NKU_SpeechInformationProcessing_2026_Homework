CONFIG=$1
GPUS=$2
MODEL_NAME=$(basename "$(dirname $CONFIG)")

# 可以使用预训练的中文模型进行微调
pretrain_G="<path to G_325000.pth>"
pretrain_D="<path to D_325000.pth>"
pretrain_dur="<path to DUR_325000.pth>"

PORT=10902

while : # auto-resume: the code sometimes crash due to bug of gloo on some gpus
do
torchrun --nproc_per_node=$GPUS \
        --master_port=$PORT \
    train.py --c $CONFIG --model $MODEL_NAME --pretrain_G $pretrain_G --pretrain_D $pretrain_D --pretrain_dur $pretrain_dur

for PID in $(ps -aux | grep $CONFIG | grep python | awk '{print $2}')
do
    echo $PID
    kill -9 $PID
done
sleep 30
done