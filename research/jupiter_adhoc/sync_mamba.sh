#!/bin/bash
PASS="${CLUSTER_SSH_PASSWORD}"
HOST="nick@winnode"
SRC="/home/jupiter/Lvl3Quant/data/processed/mbo_tensors_mamba/"
DEST="C:/Users/nick/Lvl3Quant/data/processed/mbo_tensors_mamba/"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=30 -o PasswordAuthentication=yes -o PubkeyAuthentication=no"

sshpass -p "$PASS" ssh $SSH_OPTS $HOST mkdir -p "$DEST"
echo "DIR_OK"

N=$(ls "$SRC" | wc -l)
echo "Syncing $N files to Uranus..."

sshpass -p "$PASS" scp -o StrictHostKeyChecking=no -o PasswordAuthentication=yes -o PubkeyAuthentication=no "$SRC"*.pt "$HOST:$DEST"

echo "SYNC_COMPLETE $(ls "$SRC" | wc -l) files"
