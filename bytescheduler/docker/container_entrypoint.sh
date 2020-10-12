#!/bin/bash
#
# This script is called on container startup.

CLUSTER_USER=cluster

echo "starting container services..."

if [ ! -d /run/sshd ]; then
    echo "creating /run/sshd"
    mkdir -p /run/sshd
fi

echo "starting SSHD."
/usr/sbin/sshd -f /home/$CLUSTER_USER/.ssh/sshd_config

sleep infinity
