#!/usr/bin/env bash
cd /opt/trains || exit
docker-compose -f docker-compose.yml up
