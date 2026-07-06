#!/bin/bash

# 多进程/多卡训练：ipc=host 避免 /dev/shm 不足导致 DataLoader 卡死；
# memlock/stack 与 NCCL 环境变量减轻分布式/多线程通信挂起（单机多卡也常见）。
docker run -d \
--name jhcv_lib \
--restart unless-stopped \
-p 2280:22 \
--ipc=host \
--ulimit memlock=-1 \
--ulimit stack=67108864 \
-e NCCL_P2P_DISABLE=1 \
-e NCCL_IB_DISABLE=1 \
-e NCCL_DEBUG=WARN \
-v /home/ubuntu:/home/ubuntu \
-v /data1:/data1 \
-v /etc/localtime:/etc/localtime:ro \
-e DISPLAY=:10.0 \
--env="DISPLAY" \
--privileged \
--env="QT_X11_NO_MITSHM=1" \
--gpus all \
--shm-size="64g" \
jhcv_lib:latest \
bash -c "
  # 固定 root 密码，防止镜像内部脚本修改（可根据需要修改密码）
  echo 'root:123' | chpasswd 2>/dev/null || true
  
  # 启动SSH服务
  service ssh start
  
  # 保持容器运行
  tail -f /dev/null
"