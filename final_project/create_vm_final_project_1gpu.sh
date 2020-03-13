#!/bin/bash

REGION=us-west1
ZONE=us-west1-b
GPU_TYPE=nvidia-tesla-k80
GPU_NUM=1
NAME=gpu1-project

# Config zone.
gcloud config set compute/zone ${ZONE}

gcloud compute instances create ${NAME} \
    --machine-type n1-standard-4 \
    --accelerator type=${GPU_TYPE},count=${GPU_NUM} \
    --image-family ubuntu-1804-lts --image-project ubuntu-os-cloud \
    --boot-disk-size 40GB \
    --maintenance-policy TERMINATE --restart-on-failure \
    --metadata startup-script='#!/bin/bash

    # Check for CUDA and try to install.
    # Determine whether or not we are first-time logging in by existence of CUDA
    if ! dpkg-query -W cuda; then
      # Disable ssh
      service ssh stop

      echo "[STARTUP SCRIPT] apt-get update"  
      apt-get update

      # Install openmpi
      echo "[STARTUP SCRIPT] Installing MPI"
      apt-get install openmpi-bin openmpi-common libopenmpi-dev -y

      # Install make and gcc
      echo "[STARTUP SCRIPT] Installing make and gcc"      
      apt-get install make gcc g++ -y

      # Install ruby and rmate
      echo "[STARTUP SCRIPT] Installing ruby and rmate"       
      apt install ruby -y && gem install rmate

      # Install cmake
      echo "[STARTUP SCRIPT] Installing cmake"       
      apt-get install cmake -y

      # install jre for nvvp
      echo "[STARTUP SCRIPT] Installing JRE"         
      apt-get install openjdk-8-jre -y

      # Clean up 
      echo "[STARTUP SCRIPT] apt autoremove"
      apt autoremove -y      

      # install CUDA
      echo "[STARTUP SCRIPT] Installing CUDA" 
      curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.2.89-1_amd64.deb
      dpkg -i ./cuda-repo-ubuntu1804_10.2.89-1_amd64.deb
      apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
      echo "[STARTUP SCRIPT] apt-get update"      
      apt-get update
      echo "[STARTUP SCRIPT] install CUDA"                
      apt-get install cuda -y
      rm -f ./cuda-repo*.deb

      echo "[STARTUP SCRIPT] Setting up /etc/profile for CUDA"
      echo >> /etc/profile
      echo "# CUDA" >> /etc/profile
      echo "export PATH=/usr/local/cuda-10.2/bin:/usr/local/cuda-10.2/NsightCompute-2019.1${PATH:+:${PATH}}" >> /etc/profile
      echo "export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
" >> /etc/profile

      # Download datasets
      TRAIN_IMG_FILE_NAME="train-images-idx3-ubyte.gz"
      TRAIN_LABELS_FILE_NAME="train-labels-idx1-ubyte.gz"
      TEST_IMG_FILE_NAME="t10k-images-idx3-ubyte.gz"

      mkdir data
      cd data

      echo "[STARTUP SCRIPT] Downloading training images"
      while true; do
        wget -O "$TRAIN_IMG_FILE_NAME" -c "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz" && break
      done
      gunzip "$TRAIN_IMG_FILE_NAME"

      echo "[STARTUP SCRIPT] Downloading training labels"
      while true; do
        wget -O "$TRAIN_LABELS_FILE_NAME" -c "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz" && break
      done
      gunzip "$TRAIN_LABELS_FILE_NAME"

      echo "[STARTUP SCRIPT] Downloading test images"
      while true; do
        wget -O "$TEST_IMG_FILE_NAME" -c "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz" && break
      done
      gunzip "$TEST_IMG_FILE_NAME"

      cd ..

      # Install lapack before Armadillo so Armadillo can use lapack.
      echo "[STARTUP SCRIPT] Installing LAPACK"      
      apt install liblapack-dev -y

      # Install Armadillo
      echo "[STARTUP SCRIPT] Downloading Armadillo Library"
      wget -O armadillo-7.800.2.tar.xz http://sourceforge.net/projects/arma/files/armadillo-7.800.2.tar.xz
      tar xvfJ armadillo-7.800.2.tar.xz
      cd armadillo-7.800.2
      ./configure
      make
      make install
      cd ..
      rm -rf armadillo-7.800.2*

      # Installing files for the final project start up code
      echo "[STARTUP SCRIPT] Setting the script to install the start up code"
      echo "#!/bin/bash" > /usr/bin/download_starter_code
      echo "wget https://github.com/stanford-cme213/stanford-cme213.github.io/raw/master/Code/final_project.zip" >> /usr/bin/download_starter_code
      echo "unzip final_project.zip" >> /usr/bin/download_starter_code
      echo "rm final_project.zip" >> /usr/bin/download_starter_code
      chmod ugo+x /usr/bin/download_starter_code

      # Clean up 
      echo "[STARTUP SCRIPT] apt autoremove"
      apt autoremove -y

      # Setup auto shutdown
      echo "[STARTUP SCRIPT] shutdown script"      
      wget -O /bin/auto_shutdown.sh https://raw.githubusercontent.com/stanford-cme213/stanford-cme213.github.io/master/gcp/auto_shutdown.sh
      chmod +x /bin/auto_shutdown.sh

      wget -O /etc/init.d/auto_shutdown https://raw.githubusercontent.com/stanford-cme213/stanford-cme213.github.io/master/gcp/auto_shutdown
      chmod +x /etc/init.d/auto_shutdown

      update-rc.d auto_shutdown defaults
      update-rc.d auto_shutdown enable
      service auto_shutdown start

      # Enable ssh
      service ssh start
    fi
    '

echo "Installing necessary libraries. You will be able to log into the VM after several minutes with:
$ gcloud compute ssh ${NAME}
The installation takes time. We are installing MPI, CUDA, and downloading all the files needed for the project.
After logging on the instance, please run on the VM
$ download_starter_code
This will download some source files in final_project/ to get you started.
The MNIST data files are in /data.
You can check the status of your instance at 
  https://console.cloud.google.com/compute/instances
Please check 
  https://console.cloud.google.com/billing 
to see how many credits you have left. Make sure to ask for extra credits in advance, if needed."
