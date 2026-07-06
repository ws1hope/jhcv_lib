#!/bin/bash

# 基于 jhcv_lib:deps 依赖镜像启动容器
# 相比 docker_compose.sh：镜像改为 jhcv_lib:deps，容器名/端口改掉以避免与原 jhcv_lib 容器冲突
# 3rd_party 已烤进镜像 /app/3rd_party，-v /data1:/data1 仅用于访问代码/模型/数据集，不会遮盖 /app/3rd_party
docker run -d \
--name jhcv_app \
--restart unless-stopped \
-p 2281:22 \
--ipc=host \
--ulimit memlock=-1 \
--ulimit stack=67108864 \
-e NCCL_P2P_DISABLE=1 \
-e NCCL_IB_DISABLE=1 \
-e NCCL_DEBUG=WARN \
-v /data1:/data1 \
-v /etc/localtime:/etc/localtime:ro \
-e DISPLAY=:10.0 \
--env="DISPLAY" \
--privileged \
--env="QT_X11_NO_MITSHM=1" \
--gpus all \
--shm-size="64g" \
jhcv_lib:deps \
bash -c "
  # 固定 root 密码，防止镜像内部脚本修改（可根据需要修改密码）
  echo 'root:123' | chpasswd 2>/dev/null || true

  # 启动SSH服务
  service ssh start

  # 保持容器运行
  tail -f /dev/null
"
