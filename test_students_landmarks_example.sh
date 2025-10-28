#!/bin/bash
echo 'Using Docker to start the container and run tests ...'
sudo docker build --force-rm --build-arg SSH_PRIVATE_KEY="$(cat ~/.ssh/id_rsa)" -t students_landmarks_image .
sudo docker volume create --name students_landmarks_volume
sudo docker run --name students_landmarks_container -v students_landmarks_volume:/home/username/students_landmarks --rm --gpus all -it -d students_landmarks_image bash
sudo docker exec -w /home/username/students_landmarks students_landmarks_container python test/students_landmarks_test.py --input-data test/example.tif --database wflw --gpu 0 --regressor encoder --backbone resnet50 --save-image
sudo docker stop students_landmarks_container
echo 'Transferring data from docker container to your local machine ...'
mkdir -p output
sudo chown -R "${USER}":"${USER}" /var/lib/docker/
rsync --delete -azvv /var/lib/docker/volumes/students_landmarks_volume/_data/conda/envs/students/lib/python3.10/site-packages/images_framework/output/images/ output
sudo docker system prune --all --force --volumes
sudo docker volume rm $(sudo docker volume ls -qf dangling=true)
