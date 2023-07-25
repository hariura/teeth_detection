#1/bin/bash
. ~/anaconda3/etc/profile.d/conda.sh

conda activate teeth_3

cd '/home/jh/Desktop/github/teeth_detection/prob_no_focal_sse_bb_origin/no_focal_sse_conf_sse_bb_lip_conf'
ipython print_score_prob_1-Copy1.py

cd '/home/jh/Desktop/github/teeth_detection/prob_no_focal_sse_bb_origin/no_focal_sse_conf_sse_bb_lip_no_conf'
ipython print_score_prob_1-Copy1.py

cd '/home/jh/Desktop/github/teeth_detection/prob_no_focal_sse_bb_origin/no_focal_sse_conf_sse_bb_no_lip_conf_2'
ipython print_score_prob_1-Copy1.py

cd '/home/jh/Desktop/github/teeth_detection/prob_no_focal_sse_bb_origin/no_focal_sse_conf_sse_bb_no_lip_no_conf'
ipython print_score_prob_1-Copy1.py






