#!/bin/bash -ex
model=$1

. ~/ENV/bin/activate
./train.py --load $model --evaluate
deactivate 

. ~/ENV2/bin/activate
python2 eval_oar.py
deactivate
