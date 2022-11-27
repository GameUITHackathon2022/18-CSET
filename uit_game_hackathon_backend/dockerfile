FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04
WORKDIR /base

RUN ln -sf /usr/share/zoneinfo/Asia/Ho_Chi_Minh /etc/localtime

RUN apt-get update && apt-get install -y python3 python3-pip cmake wget llvm
RUN apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1-mesa-glx
RUN pip3 install numpy==1.16.1
RUN pip3 install opencv-python==4.2.0.32
RUN pip3 install configparser==5.0.2
RUN pip3 install six==1.16.0
RUN pip3 install future==0.18.2
RUN pip3 install python-multipart
RUN pip3 install uvicorn[standard] fastapi==0.65.2
RUN pip3 install requests
RUN pip3 install pillow==8.0.0
RUN pip3 install multipledispatch
RUN pip3 install imutils pymysql
RUN pip3 install easydict
RUN pip3 install cryptography
RUN pip3 install WebSockets
CMD  sh start_service.sh

