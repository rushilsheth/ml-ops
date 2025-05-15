copied from this amazing lab: https://github.com/mlip-cmu/s2025/blob/main/labs/lab09.md

## Setup Our Server

### 1. Build and Push the Backend Image
```bash
docker build -t rushilsheth/full_lab:1.0.0 -f Dockerfile.backend .
docker push rushilsheth/full_lab:1.0.0
```

### 2. Deploy the Backend
Apply the backend deployment and service manifests:
```bash
kubectl apply -f backend-deployment.yaml
kubectl apply -f backend-service.yaml
```

## Setup Our Load Balancer

### 1. Build and Push the Load Balancer Image
```bash
docker build -t rushilsheth/load_balancer_lab:1.0.0 -f Dockerfile.loadbalancer .
docker push rushilsheth/load_balancer_lab:1.0.0
```

### 2. Deploy the Load Balancer
Apply the load balancer deployment and service manifests:
```bash
kubectl apply -f loadbalancer-deployment.yaml
kubectl apply -f loadbalancer-service.yaml
```

## Setup a Tunnel (or Switch to VirtualBox)

### Option A: Use a Tunnel with Docker Driver
If you are using the Docker driver on macOS, NodePort services may not be directly accessible. Run a tunnel in a separate terminal window:
```bash
minikube start
minikube service flask-load-balancer-service
```
Keep this terminal running; it will expose your NodePort service externally.

### Option B: Switch to VirtualBox Driver
Alternatively, you can delete your current Minikube cluster and start a new one using VirtualBox:
```bash
minikube delete
minikube start --driver=virtualbox
minikube ip
```
Using VirtualBox makes NodePort services directly accessible via the Minikube IP without needing a tunnel.

## Test Invocation
Once everything is deployed and running, test your load balancer service. For example, if the Minikube IP is `192.168.99.100` and your NodePort is `30080`:
```bash
curl "http://127.0.0.1:62494/?user_id=Alice"
```
Replace the IP with the output from:
```bash
minikube ip
```

Enjoy!