# ECE522 MP4

## 0. Hardware and Software Dependencies

The artifact can be executed on any x86 machine with at least 30 GB of main memory and at least 40 GB of disk space. We strongly recommend running the artifact on a workstation with multi-cores and at least 128 GB memory. The artifact needs a Linux environment (preferably Ubuntu) and a compiler that supports the C++14 standard.

## 1. Installation

### 1.1 Downloading the Repository
Use the following command to download the artifact:

```bash
git clone git@github.com:0-EricZhou-0/ece522-mp4.git
```

### 1.2 Installation

Install the following dependencies:
```bash
sudo apt install flex bison tmux python3-pip
```

Build simulator (the output executable is named `sim`):
```bash
cd src
make clean
make -j
```

## 2. Experiment Workflow

### 2.1 Launching A Single Experiment
Every configuration file specifies the DNN model and the batch size to be used, as well as other system configuration parameters (such as GPU memory size, SSD Bandwidth, the baseline type, and so on).

To run a single experiment, directly find its corresponding config file and use. For example:
```bash
./sim "$relative_path_to_config_file"
    # e.g.,  ./sim configs/example.config
```

