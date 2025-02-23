#!/bin/bash
eval $(minikube docker-env)
docker build -t face-webrtc-app .
#minikube image load face-webrtc-app:latest
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
#kubectl port-forward service/my-face-webrtc-app 7860:7080
