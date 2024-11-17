#!/bin/bash

gcloud_vmname="z101"
gcloud_zone="us-west4-a"
gcloud_instancetemp="temp-08vcpu-128ram"
gcloud_user="fililoco"

# Definir la ruta del archivo de configuración como una variable de entorno
config_path="/home/$gcloud_user/dmeyf2024/kaggle2/config_competencia_02_lgmb_z101.yaml"
gcloud compute ssh $gcloud_user@$gcloud_vmname --zone $gcloud_zone --command="
    export CONFIG_PATH=$config_path &&
    cd /home/$gcloud_user/buckets/b1/public/gcloud/ &&
    source /home/$gcloud_user/.venv/bin/activate &&
    /home/fililoco/buckets/b1/public/reposync.sh &&
    python3 /home/fililoco/dmeyf2024/kaggle2/handler_optuna.py
    "
