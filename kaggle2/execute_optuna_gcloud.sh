#!/bin/bash

gcloud_vmname="z101"
gcloud_zone="us-west4-a"
gcloud_instancetemp="temp-08vcpu-128ram"
gcloud_user="fililoco"

gcloud compute ssh $gcloud_user@$gcloud_vmname --zone us-west4-a --command="
    cd /home/$gcloud_user/buckets/b1/public/gcloud/ &&
    source /home/$gcloud_user/.venv/bin/activate &&
    /home/fililoco/buckets/b1/public/reposync.sh &&
    python3 /home/fililoco/buckets/b1/repos/dmeyf2024/kaggle2/handler_optuna.py
    "