#!/bin/bash
#definirr las variables
gcloud_vmname="z101"
gcloud_zone="us-west4-a"
gcloud_instancetemp="temp-08vcpu-128ram"

# Ejecutar el comando gcloud
gcloud compute instances create "$gcloud_vmname" \
    --source-instance-template="$gcloud_instancetemp" \
    --zone="$gcloud_zone"