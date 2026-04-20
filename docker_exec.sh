#!/bin/bash
xhost +local:root
docker start 5cdd1d8f092e 
docker exec -it 5cdd1d8f092e bash

