#!/bin/bash
xhost +local:root
docker start 9a57e4b7e985 
docker exec -it 9a57e4b7e985 bash

