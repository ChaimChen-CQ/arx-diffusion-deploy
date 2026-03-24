FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata locales curl gnupg2 lsb-release ca-certificates \
    build-essential git vim wget unzip \
    python3 python3-pip python3-dev python3-venv \
    iputils-ping net-tools usbutils udev \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

RUN locale-gen en_US en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

RUN echo "deb http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros1.list && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -

RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-desktop-full \
    python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool \
    python3-catkin-tools python3-osrf-pycommon \
    && rm -rf /var/lib/apt/lists/*

RUN rosdep init || true
RUN rosdep update

RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc

WORKDIR /workspace/genrobot_controller_sdk

CMD ["/bin/bash"]
