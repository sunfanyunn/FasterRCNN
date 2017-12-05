#!/bin/bash -ex
model=train_log/fastrcnn/model-79800

. ~/ENV/bin/activate
./train.py --load $model --evaluate
deactivate 

. ~/ENV2/bin/activate
python2 eval_oar.py
deactivate
