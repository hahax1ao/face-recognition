首先安装minikube（此为MacOS，若为其他操作系统，请参考官方文档：https://minikube.sigs.k8s.io/docs/start/?arch=%2Fmacos%2Fx86-64%2Fstable%2Fbinary+download#Service）
curl -LO https://github.com/kubernetes/minikube/releases/latest/download/minikube-darwin-amd64
sudo install minikube-darwin-amd64 /usr/local/bin/minikube

minikube start （需要比较长的时间，若无法下载镜像，可以使用华为云：docker pull swr.cn-north-4.myhuaweicloud.com/ddn-k8s/gcr.io/k8s-minikube/kicbase:v0.0.46）
通过以下语句验证
kubectl get po -A


执行start.sh
bash start.sh
若拉取python：3.12失败，则手动拉取
pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
以上这一步需要比较久的时间，预计10分钟
打包好的镜像叫做 face-webrtc-app

通过以下指令查看
eval $(minikube docker-env)
docker images
将镜像推送到minikube的镜像仓库
minikube cache add face-webrtc-app:latest
如果镜像发生更改，执行minikube cache reload


接下来是minikube的操作创建服务
kubectl apply -f deployment.yaml
#kubectl create deployment my-face-webrtc-app --image=face-webrtc-app:latest
#kubectl expose deployment my-face-webrtc-app --type=NodePort --port=7860
#kubectl delete deployment my-face-webrtc-app

使用此命令检查是否完成
kubectl apply -f service.yaml
#kubectl get services my-face-webrtc-app
#kubectl delete service my-face-webrtc-app

访问服务
kubectl get svc my-face-webrtc-app

minikube service my-face-webrtc-app

若镜像获取不到，可以手动加载镜像
minikube image load face-webrtc-app:latest
如果镜像有更新，需要重新加载更新，并且删除pod，让服务自动拉取最新镜像
kubectl delete pod --selector=app=face-webrtc-app
或者直接删除重新创建镜像
docker rmi face-webrtc-app:latest
docker build -t face-webrtc-app:latest .
minikube image load face-webrtc-app:latest


执行此命令查看看板
minikube dashboard

一切正常后，执行此指令发布服务
kubectl port-forward service/my-face-webrtc-app 7860:7080

访问：127.0.0.1:7860即可


