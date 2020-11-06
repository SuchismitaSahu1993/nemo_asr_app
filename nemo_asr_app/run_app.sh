#!/bin/bash
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
################################################################################
# Run Application
################################################################################
INPUT_PORT=8888

# check if arguments supplied
if [ $# -eq 0 ]; then  # no arguments - ask for port and data dir
    # data dir
    read -e -p "Please enter the path to the data directory: " DATA_DIR
    # port
    read -e -p "Enter the port to run the container's jupyter notebooks (Press ENTER to use default port $INPUT_PORT):" NEW_INPUT_PORT
    if [ "$NEW_INPUT_PORT" ]; then
	INPUT_PORT=$NEW_INPUT_PORT
    fi
elif [ $# -eq 2 ]; then
     # arguments provide port and data dir
     INPUT_PORT=$1
     DATA_DIR=$2
else
    echo "Provide no arguments or 2 arguments: port and data_dir."
    exit
fi

# container and image names
DATA_DIR="${DATA_DIR/#\~/$HOME}"
IMAGE_NAME='nemo_asr_app_ngc'
USERNAME=`whoami`
CONTAINER_NAME='run_nemo_asr_app_cont_'$USERNAME

# host IP for remote access 
HOST_IP=`hostname -I | awk '{print $1;}'`

# check port is not in use
while true; do
	(echo >/dev/tcp/localhost/$INPUT_PORT) &>/dev/null && echo "TCP port $INPUT_PORT is in use." || break
	read -e -p "Please enter a different port:" INPUT_PORT
done

echo "Using $DOCKER_CMD"
echo "Container: $CONTAINER_NAME"
echo "Image: $IMAGE_NAME"
echo "Port: $INPUT_PORT"
echo "Data directory: $DATA_DIR"

# Build the application docker file
echo "Building image from Dockerfile"
docker build -t $IMAGE_NAME .

# Clean out old containers
docker rm $(docker stop $(docker ps -a -q --filter status=exited)) 2>/dev/null
docker rm $(docker stop $(docker ps -a -q --filter ancestor=$CONTAINER_NAME))
docker rmi $(docker images -q -f "dangling=true") 2>/dev/null

# Run command
APP_DIR=/home/ssahu/nemo_asr_app # path to application

docker run --gpus all -it --rm --name $CONTAINER_NAME \
            --ipc=host \
	    --env DATA_DIR=$DATA_DIR \
	    -v $DATA_DIR:$DATA_DIR \
	    -v $APP_DIR:$APP_DIR \
	    -v $APP_DIR:/home/ssahu/nemo_asr_app/Suchi \
            -p $INPUT_PORT:8888 \
            $IMAGE_NAME 

# Clean up
find $APP_DIR -name \*.pyc -delete
echo "Done with ${IMAGE_NAME} service run"
