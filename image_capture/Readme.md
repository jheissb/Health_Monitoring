
# Installation:

## First things first...
```sh
sudo apt update
sudo apt upgrade
sudo pip3 install --upgrade pip
sudo reboot
```

## Install CSI Arducam 12MP

OS system must be on the SD card for the driver to work. After installation, the OS can be moved to SSD drive.

1- Make sure you have L4T32.4.3 or newer. I reinstalled to Jetpack 4.5 on Jetson NX to make sure it works.

2- Go to https://www.arducam.com/docs/camera-for-jetson-nano/native-jetson-cameras-imx219-imx477/imx477-how-to-install-the-driver/

Make sure during boot that the first message reads MX477 and not MX219.

3- Connect the camera to the CSI port. Follow instructions to download and install driver and reboot

4- Test camera by running their script (from https://www.arducam.com/docs/camera-for-jetson-nano/native-jetson-cameras-imx219-imx477/imx477/#9-display):

```
SENSOR_ID=0 # 0 for CAM0 and 1 for CAM1 ports
FRAMERATE=60 # Framerate can go from 2 to 60 for 1920x1080 mode
gst-launch-1.0 nvarguscamerasrc sensor-id=$SENSOR_ID ! "video/x-raw(memory:NVMM),width=1920,height=1080,framerate=$FRAMERATE/1" ! nvvidconv ! nvoverlaysink
```

Exit with ctrl-C



## Install TRT_pose on Jetson NX

(see https://github.com/NVIDIA-AI-IOT/trt_pose for details)

* Install pytorch  1.7.0
```sh
wget https://nvidia.box.com/shared/static/wa34qwrwtk9njtyarwt5nvo6imenfy26.whl -O torch-1.7.0-cp36-cp36m-linux_aarch64.whl
sudo apt-get install python3-pip libopenblas-base libopenmpi-dev 
pip3 install Cython
pip3 install numpy torch-1.7.0-cp36-cp36m-linux_aarch64.whl
```

* install pillow 
```sh
sudo apt install libjpeg8-dev zlib1g-dev libtiff-dev libfreetype6 libfreetype6-dev libwebp-dev libopenjp2-7-dev libopenjp2-7-dev -y
pip3 install pillow --global-option="build_ext" \
--global-option="--enable-zlib" \
--global-option="--enable-jpeg" \
--global-option="--enable-tiff" \
--global-option="--enable-freetype" \
--global-option="--enable-webp" \
--global-option="--enable-webpmux" \
--global-option="--enable-jpeg2000"
```

* Install torchvision 0.8.1(*must be compatible with pytorch version*)
```sh
sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev
git clone --branch v0.8.1 https://github.com/pytorch/vision torchvision 
cd torchvision
export BUILD_VERSION=0.8.1  # where 0.x.0 is the torchvision version  
python3 setup.py install --user
cd ../  # attempting to load torc
```
IF this doesn't work, try building it from here:

https://github.com/pytorch/vision

Or here:
https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-9-0-now-available/72048


* Install torch2trt
```sh
cd /usr/local/src
sudo git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt
sudo python3 setup.py install
```

* Install program 
```sh
pip3 install tqdm cython pycocotools
sudo apt-get install python3-matplotlib
cd /usr/local/src
sudo git clone https://github.com/NVIDIA-AI-IOT/trt_pose
cd trt_pose
sudo python3 setup.py install
```

### Download TRTpose model

Within the trt_pose folder, download mdel from below and place it in the tasks/human_pose directory

resnet18_baseline_att_224x224_A:
https://drive.google.com/open?id=1XYDdCUdiF2xxx4rznmLb62SdOUZuoNbd

densenet121_baseline_att_256x256_B:
https://drive.google.com/open?id=13FkJkx7evQ1WwP54UmdiDXWyFMY1OxDU


## Install Yolo5 on Jetson NX
```sh
git clone https://github.com/ultralytics/yolov5

cd yolov5

sudo apt update && sudo apt install -y libffi-dev python3-pip curl unzip python3-tk libopencv-dev python3-opencv 
sudo apt install -y python3-scipy python3-matplotlib python3-numpy
sudo pip3 install git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI
sudo pip3 install scikit-learn==0.19.2
sudo pip3 install seaborn
sudo pip3 install -r requirements.txt
```

If there are error messages, run the test below any way as it will download any missing coimponents

### Test: Connect usb camera to Jetson and run:
```sh
MODEL=yolov5m.pt
CAM=0
python3 detect.py --source $CAM --weights $MODEL --conf 0.5

close with ctrl-C
```

## Install pyqt5:
```sh
pip3 install PyQt5-sip
sudo apt-get install qt5-default pyqt5-dev pyqt5-dev-tools
pip3 install --upgrade setuptools
sudo apt-get install qttools5-dev-tools
```

# Image Capture
```sh
git clone https://github.com/yangrong0830/w210_capstone.git
cd /w210_capstone/image_capture/GUI
Copy the downloaded trt-pose models also inside this folder

## Run Image capture:
```sh
cd /w210_capstone/image_capture/GUI
python3 wwatchers2021.py
```
