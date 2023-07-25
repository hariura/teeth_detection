#1/bin/bash
. ~/anaconda3/etc/profile.d/conda.sh

conda activate teeth_3



cd '/home/jh/Desktop/github/teeth_detection/test8/teeth_training_lip_conf' 
python train_config.py

cd '/home/jh/Desktop/github/teeth_detection/test8/teeth_training_lip_no_conf'
python train_config.py

cd '/home/jh/Desktop/github/teeth_detection/test8/teeth_training_no_lip_config' 
python train_config.py

cd '/home/jh/Desktop/github/teeth_detection/test8/teeth_training_no_lip_no_conf' 
python train_config.py






