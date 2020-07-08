#!/usr/bin/env bash

echo "vm.max_map_count=262144" > /tmp/99-trains.conf
sudo mv /tmp/99-trains.conf /etc/sysctl.d/99-trains.conf
sudo sysctl -w vm.max_map_count=262144
sudo service docker restart

sudo mkdir -p /opt/trains/data/elastic
sudo mkdir -p /opt/trains/data/mongo/db
sudo mkdir -p /opt/trains/data/mongo/configdb
sudo mkdir -p /opt/trains/data/redis
sudo mkdir -p /opt/trains/logs
sudo mkdir -p /opt/trains/config
sudo mkdir -p /opt/trains/data/fileserver

sudo chown -R 1000:1000 /opt/trains

cd /opt/trains || exit
curl https://raw.githubusercontent.com/allegroai/trains-server/master/docker-compose.yml -o docker-compose.yml
