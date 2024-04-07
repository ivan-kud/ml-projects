###############################################
# A f t e r   D r o p l e t   c r e a t i o n #
###############################################
# Change DNS records to new IP

# SSH to host as 'root'
ssh root@ivankud.com
# or (if DNS records is still not updated)
ssh root@<IP-address>

# Install updates and restart droplet (optional)
apt update && apt upgrade -y
shutdown -r now

# Setup wireguard (optional, see section below)

# Clone repository
git clone https://github.com/ivan-kud/ml-service.git

# Add environment variables
export TRAEFIK_USERNAME=<USERNAME>
export TRAEFIK_PASSWORD=<PASSWORD>
export TRAEFIK_HASHED_PASSWORD=$(openssl passwd -apr1 $TRAEFIK_PASSWORD)

# Create network
docker network create traefik-public

# Build images and start containers (use -d option to detach terminal)
cd ml-service
docker compose up


#################################
# S e t u p   W i r e G u a r d #
#################################
apt install -y wireguard
wg genkey | tee /etc/wireguard/private.key | wg pubkey | tee /etc/wireguard/public.key
chmod 600 /etc/wireguard/private.key
nano /etc/wireguard/wg0.conf
[Interface]
PrivateKey = <server_private_key>
Address = 10.0.0.1/24
ListenPort = 51820
PostUp = iptables -A FORWARD -i %i -j ACCEPT; iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE
PostDown = iptables -D FORWARD -i %i -j ACCEPT; iptables -t nat -D POSTROUTING -o eth0 -j MASQUERADE
echo "net.ipv4.ip_forward=1" >> /etc/sysctl.conf
sysctl -p
ufw allow 51820/udp
ufw disable
ufw enable
systemctl enable wg-quick@wg0.service
systemctl start wg-quick@wg0.service
systemctl status wg-quick@wg0.service

# Add configuration for client devices
wg genkey | tee /etc/wireguard/macbook_private.key | wg pubkey | tee /etc/wireguard/macbook_public.key
wg genkey | tee /etc/wireguard/android_private.key | wg pubkey | tee /etc/wireguard/android_public.key
nano /etc/wireguard/wg0.conf
[Peer]
PublicKey = <macbook_client_public_key>
AllowedIPs = 10.0.0.2/32

[Peer]
PublicKey = <android_client_public_key>
AllowedIPs = 10.0.0.3/32
systemctl restart wg-quick@wg0.service
systemctl status wg-quick@wg0.service

# Add configuration to clients
[Interface]
PrivateKey = <client_private_key>
Address = <client_VPN_IP_address>/32
DNS = 8.8.8.8

[Peer]
PublicKey = <server_public_key>
Endpoint = <server_IP_address>:51820
AllowedIPs = 0.0.0.0/0
PersistentKeepalive = 20


#######################################
# D e p l o y   n e w   v e r s i o n #
#######################################
ssh root@ivankud.com

# create envs
export TRAEFIK_USERNAME=<USERNAME>
export TRAEFIK_PASSWORD=<PASSWORD>
export TRAEFIK_HASHED_PASSWORD=$(openssl passwd -apr1 $TRAEFIK_PASSWORD)

# Stop containers
cd ml-service
docker compose stop
cd ..

# Update source files
rm -rf ml-service
git clone https://github.com/ivan-kud/ml-service.git

# Rebuild image
cd ml-service
docker build . -t ml-service-fastapi

# Delete old container and image
docker container ls -a
docker rm <FASTAPI_CONTAINER_ID>
docker images
docker rmi <FASTAPI_IMAGE_ID>

# Start containers (use -d option to detach terminal)
docker compose up


#################################
# U s e f u l   C o m m a n d s #
#################################
# Run application locally
cd app
uvicorn src.main:app --reload

# Build docker image
docker build . -f Dockerfile-locally -t ivankud/ml-service
# Run docker container
docker run --rm -it -p 80:80 ivankud/ml-service
# Login to Docker Hub
docker login -u ivankud
# If needed, tag your image with your registry username
docker tag ml-service-fastapi ivankud/ml-service
# Push image to Docker Hub repository
docker push ivankud/ml-service
# Pull image form Docker Hub repository
docker pull ivankud/ml-service:latest

# Docker Compose
docker compose -f compose-locally.yaml up -d
docker compose ps
docker compose logs
docker compose pause
docker compose unpause
docker compose stop
docker compose start
docker compose -f compose-locally.yaml down

# Examine the list of installed UFW profiles
ufw app list
# Allow SSH connections and ports
ufw allow OpenSSH
ufw allow 80
ufw allow 443
# Enable the firewall
ufw enable
# See connections that are still allowed
ufw status

# Add non-root user 'ml-service'
adduser ml-service
# Add your new user to the sudo group
usermod -aG sudo ml-service
# Copy your local public SSH key to new user to log in with SSH
rsync -a --chown=ml-service:ml-service ~/.ssh /home/ml-service
# Now, open up a new terminal session, and use SSH to log in as new user
ssh ml-service@ivankud.com

# Find and pull certificates to local machine (command should be executed on the local machine)
find / -type f -iname "acme.json"
rsync -a root@ivankud.com:/var/lib/docker/volumes/ml-service_traefik-public-certificates/_data/ ~/Documents/GitHub/ml-service/certificates

# Run JupyterLab
jupyter-lab

# Run TensorBoard
tensorboard --logdir=runs

pip install 'huggingface_hub[cli]'
huggingface-cli delete-cache
