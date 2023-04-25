TRAINSF(){
pretrain_name=BidNet
cd ..
mkdir logs
loss=config/loss_config_disp.json
outf_model=models_saved/$pretrain_name
logf=logs/$pretrain_name
datapath=/media/zliu/datagrid1/liu/sceneflow
datathread=4
lr=1e-3
devices=0
dataset=sceneflow
trainlist=filenames/SceneFlow_Fog.list
vallist=filenames/SceneFlow_Fog_Val.list
startR=0
startE=0
batchSize=4
testbatch=1
maxdisp=-1
save_logdir=experiments_logdir/$pretrain_name
model=$pretrain_name
pretrain=none
initial_pretrain=none

python3 -W ignore train_BidNet.py --cuda --loss $loss --lr $lr \
               --outf $outf_model --logFile $logf \
               --devices $devices --batch_size $batchSize \
               --dataset $dataset --trainlist $trainlist --vallist $vallist \
               --startRound $startR --startEpoch $startE \
               --model $model \
               --maxdisp $maxdisp \
               --datapath $datapath \
               --manualSeed 1024 --test_batch $testbatch \
               --save_logdir $save_logdir \
               --pretrain $pretrain \
               --initial_pretrain $initial_pretrain
}


TRAINSF