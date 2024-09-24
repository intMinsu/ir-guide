#!/bin/bash
# https://qengineering.eu/install-opencv-on-jetson-nano.html

version="4.10.0"
FILE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
folder="${FILE_DIR}/../opencv-workspace"
echo "opencv-${version} and opencv_contrib-${version} will be downloaded onto ${folder}."
echo ""

# Check if the file /proc/device-tree/model exists
if [ -e "/proc/device-tree/model" ]; then
    model=$(tr -d '\0' < /proc/device-tree/model)
    # Check if the model information contains "Jetson Nano Orion"
    echo ""
    if [[ $model == *"Orin"* ]]; then
        echo "Detecting a Jetson Nano Orin."
	  # Use always "-j 4"
        NO_JOB=4
        ARCH=8.7
        PTX="sm_87"
    elif [[ $model == *"Xavier"* ]]; then
        echo "Detecting a Jetson Xavier."
        NO_JOB=4
        ARCH="7.2,8.7"
        PTX="sm_72"
    else
        echo "We expect the device to be either Orin or Xavier."
        echo "For Jetson Nano, Refer to https://raw.githubusercontent.com/Qengineering/Install-OpenCV-Jetson-Nano/main/OpenCV-4-10-0.sh"
        exit 1
    fi
    echo ""
else
    echo "Error: /proc/device-tree/model not found. Are you sure this is a Jetson?"
    exit 1
fi


for (( ; ; ))
do
    echo "Do you want to remove the default OpenCV (yes/no)?"
    read rm_old

    if [ "$rm_old" = "yes" ]; then
        echo "** Remove other OpenCV first"
        sudo apt -y purge *libopencv*
	break
    elif [ "$rm_old" = "no" ]; then
	break
    fi
done


echo "------------------------------------"
echo "** Install requirement (1/4)"
echo "------------------------------------"
sudo apt-get update
sudo apt-get install -y build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
sudo apt-get install -y ffmpeg
sudo apt-get install -y python3.8-dev python-dev python-numpy python3-numpy
sudo apt-get install -y libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev
sudo apt-get install -y libv4l-dev v4l-utils qv4l2 v4l2ucp
sudo apt-get install -y libxvidcore-dev libx264-dev libmp3lame-dev libvorbis-dev libfaac-dev libtheora-dev
sudo apt-get install -y libopencore-amrnb-dev libopencore-amrwb-dev
sudo apt-get install -y libjpeg8-dev libjpeg-turbo8-dev libglew-dev libcanberra-gtk*
sudo apt-get install -y libtbb-dev libxine2-dev
sudo apt-get install -y libopenblas-dev libatlas-base-dev libblas-dev liblapack-dev liblapacke-dev libeigen3-dev
sudo apt-get install -y libhdf5-dev libprotobuf-dev protobuf-compiler libgoogle-glog-dev libgflags-dev
sudo apt-get install -y libtesseract-dev
sudo apt-get install -y curl


echo "------------------------------------"
echo "** Download opencv "${version}" (2/4)"
echo "------------------------------------"
mkdir $folder
cd ${folder}
curl -L https://github.com/opencv/opencv/archive/${version}.zip -o opencv-${version}.zip
curl -L https://github.com/opencv/opencv_contrib/archive/${version}.zip -o opencv_contrib-${version}.zip
unzip opencv-${version}.zip
unzip opencv_contrib-${version}.zip
rm opencv-${version}.zip opencv_contrib-${version}.zip
cd opencv-${version}/


echo "------------------------------------"
echo "** Build opencv "${version}" (3/4)"
echo "------------------------------------"
mkdir release
cd release/
cmake -D WITH_CUDA=ON \
    -D WITH_CUDNN=ON \
    -D WITH_CUFFT=ON \
    -D ENABLE_FAST_MATH=ON \
    -D CUDA_FAST_MATH=ON \
    -D OPENCV_DNN_CUDA=ON \
    -D CUDA_ARCH_BIN=${ARCH} \
    -D CUDA_ARCH_PTX=${PTX} \
    -D WITH_FFMPEG=ON \
    -D WITH_GSTREAMER=ON \
    -D WITH_V4L=ON \
    -D WITH_QT=OFF \
    -D WITH_OPENMP=ON \
    -D WITH_TBB=ON \
    -D WITH_OPENGL=ON \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-${version}/modules \
    -D BUILD_opencv_python3=ON \
    -D BUILD_TESTS=OFF \
    -D BUILD_PERF_TESTS=OFF \
    -D BUILD_EXAMPLES=OFF \
    -D INSTALL_C_EXAMPLES=OFF \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=${CONDA_PREFIX} \
    -D PYTHON3_EXECUTABLE=$(which python3) \
    -D PYTHON3_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
    -D PYTHON3_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
    -D PYTHON3_PACKAGES_PATH=$(python -c "import site; print(site.getsitepackages()[0])") \
    ..
make -j${NO_JOB} # $(nproc)=6 (Fully using procs) or ${NO_JOB}=4 as indicated in Qengineering

# More explicitly:
# -D PYTHON3_INCLUDE_DIR=${CONDA_PREFIX}/include/python3.8 \
# -D PYTHON3_PACKAGES_PATH=${CONDA_PREFIX}/lib/python3.8/site-packages \

echo "------------------------------------"
echo "** Install opencv "${version}" (4/4)"
echo "------------------------------------"
sudo make install
sudo ldconfig
#echo 'export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
#echo 'export PYTHONPATH=/usr/local/lib/python3.8/site-packages/:$PYTHONPATH' >> ~/.bashrc
#source ~/.bashrc


echo "** Install opencv "${version}" successfully"
echo "** Bye :)"
