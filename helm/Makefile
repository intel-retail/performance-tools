# Copyright © 2024 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

.PHONY: build build-platform build-igt build-xpum build-minikube run-minikube-demo stop-minikube-demo helm-convert install-helm uninstall-helm

build: build-platform build-igt build-xpum

build-platform:
	echo "Building benchmark container HTTPS_PROXY=${HTTPS_PROXY} HTTP_PROXY=${HTTP_PROXY}"
	cd platform && docker build --build-arg HTTPS_PROXY=${HTTPS_PROXY} --build-arg HTTP_PROXY=${HTTP_PROXY} -t benchmark:platform -f Dockerfile .

build-igt:
	echo "Building igt HTTPS_PROXY=${HTTPS_PROXY} HTTP_PROXY=${HTTP_PROXY}"
	cd intel-top && docker build --build-arg HTTPS_PROXY=${HTTPS_PROXY} --build-arg HTTP_PROXY=${HTTP_PROXY} -t benchmark:igt -f Dockerfile .

build-xpum:
	echo "Building xpu HTTPS_PROXY=${HTTPS_PROXY} HTTP_PROXY=${HTTP_PROXY}"
	cd xpum && docker build --build-arg HTTPS_PROXY=${HTTPS_PROXY} --build-arg HTTP_PROXY=${HTTP_PROXY} -t benchmark:xpum -f Dockerfile .

build-minikube: build 
	minikube start
	cd ./intel-top && minikube image build -t benchmark:igt -f ./Dockerfile . && cd ../
	cd ./platform && minikube image build -t benchmark:platform -f ./Dockerfile . && cd ../
	# cd ./xpum && minikube image build -t benchmark:xpum -f ./Dockerfile . && cd ../

helm-convert:
	log_dir=/tmp/results kompose -f ./docker-compose.yaml convert -o edge-video-performance-evaluation-tool  -c
	sed -i 's/version: [0-9].[0-9].[0-9]/version: 3.2.0/' ./edge-video-performance-evaluation-tool/Chart.yaml
	helm package edge-video-performance-evaluation-tool

run-minikube-demo: build-minikube
	log_dir=/tmp/results kompose -f docker-compose.yaml convert -o kubernetes/ && \
	kubectl apply -f kubernetes
	
stop-minikube-demo:
	kubectl delete -f kubernetes
	minikube stop

install-helm:
	helm install -f kubernetes/Chart.yaml dls kubernetes

uninstall-helm:
	helm uninstall dls