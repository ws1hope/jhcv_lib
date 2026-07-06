FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# 配置清华源
RUN echo "deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy main restricted universe multiverse" > /etc/apt/sources.list && \
    echo "deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-backports main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-security main restricted universe multiverse" >> /etc/apt/sources.list

# 安装基础依赖和OpenGL库
RUN apt-get update && apt-get install -y openssh-server 


# 复制SSH配置文件
COPY deploy/sshd_config /etc/ssh/sshd_config

# 配置SSH服务
RUN mkdir /var/run/sshd && \
    echo 'root:123' | chpasswd && \
    chmod 600 /etc/ssh/sshd_config && \
    LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH ssh-keygen -A

# 配置SSH环境变量
RUN echo 'export CUDA_HOME=/usr/local/cuda-11.8' >> /root/.bashrc && \
    echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> /root/.bashrc && \
    echo 'export LD_LIBRARY_PATH=/opt/miniconda3/envs/ai/lib:/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> /root/.bashrc && \
    echo 'export PATH=/opt/miniconda3/bin:$PATH' >> /root/.bashrc

# 暴露SSH端口
EXPOSE 22