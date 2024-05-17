# Set up the package repository and key
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Update the package lists
sudo apt-get update

# Install the NVIDIA container toolkit
sudo apt-get install -y nvidia-container-toolkit

# Restart the Docker daemon to load the new configuration
sudo systemctl restart docker
