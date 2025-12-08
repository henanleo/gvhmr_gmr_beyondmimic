# gvhmr_gmr_beyondmimic

This project, in conjunction with [HVGMR](https://github.com/zju3dv/GVHMR), [GMR](https://github.com/YanjieZe/GMR), and [Beyondmimic](https://github.com/HybridRobotics/whole_body_tracking), aims to create a fast robot policy training script. 

# Install

## Step1:video2csv

### Environment

```bash
git clone https://github.com/henanleo/gvhmr_gmr_beyondmimic.git
cd gvhmr_gmr_beyondmimic
conda create -y -n video2csv python=3.10
conda activate video2csv
pip install -r requirements.txt
pip install -e .
cd GMR
pip install -e .
cd..
mkdir inputs
mkdir outputs
```

**Weights**

```bash
mkdir -p inputs/checkpoints

# 1. You need to sign up for downloading [SMPL](https://smpl.is.tue.mpg.de/) and [SMPLX](https://smpl-x.is.tue.mpg.de/). And the checkpoints should be placed in the following structure:

inputs/checkpoints/
├── body_models/smplx/
│   └── SMPLX_{GENDER}.npz # SMPLX (We predict SMPLX params + evaluation)
└── body_models/smpl/
    └── SMPL_{GENDER}.pkl  # SMPL (rendering and evaluation)

GMR/assets/body_models/smplx/
├── SMPLX_NEUTRAL.pkl
├── SMPLX_FEMALE.pkl
└── SMPLX_MALE.pkl
# 2. Download other pretrained models from Google-Drive (By downloading, you agree to the corresponding licences): https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD?usp=drive_link

inputs/checkpoints/
├── dpvo/
│   └── dpvo.pth
├── gvhmr/
│   └── gvhmr_siga24_release.ckpt
├── hmr2/
│   └── epoch=10-step=25000.ckpt
├── vitpose/
│   └── vitpose-h-multi-coco.pth
└── yolo/
    └── yolov8x.pt
   
# 3 You need to download GMR/assets from https://github.com/YanjieZe/GMR.
```

## Step2:train

### Environment

```bash
cd whole_body_tracking
curl -L -o unitree_description.tar.gz https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz && \
tar -xzf unitree_description.tar.gz -C source/whole_body_tracking/whole_body_tracking/assets/ && \
rm unitree_description.tar.gz
conda create -y -n beyondmimic python=3.10
conda activate beyondmimic
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install "isaacsim[all,extscache]==4.5.0" --extra-index-url https://pypi.nvidia.com
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install
python -m pip install -e source/whole_body_tracking
```

**Weights**

```bash
mkdir -p inputs/checkpoints

# 1. You need to sign up for downloading [SMPL](https://smpl.is.tue.mpg.de/) and [SMPLX](https://smpl-x.is.tue.mpg.de/). And the checkpoints should be placed in the following structure:

inputs/checkpoints/
├── body_models/smplx/
│   └── SMPLX_{GENDER}.npz # SMPLX (We predict SMPLX params + evaluation)
└── body_models/smpl/
    └── SMPL_{GENDER}.pkl  # SMPL (rendering and evaluation)

GMR/assets/body_models/smplx/
├── SMPLX_NEUTRAL.pkl
├── SMPLX_FEMALE.pkl
└── SMPLX_MALE.pkl
# 2. Download other pretrained models from Google-Drive (By downloading, you agree to the corresponding licences): https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD?usp=drive_link

inputs/checkpoints/
├── dpvo/
│   └── dpvo.pth
├── gvhmr/
│   └── gvhmr_siga24_release.ckpt
├── hmr2/
│   └── epoch=10-step=25000.ckpt
├── vitpose/
│   └── vitpose-h-multi-coco.pth
└── yolo/
    └── yolov8x.pt
   
# 3 You need to download GMR/assets from https://github.com/YanjieZe/GMR.
```



## Usage

Step1: video2csv

```bash
conda activate video2csv
python video2csv.py --video {video}.mp4 --output_csv {output_root}
```

step2:train

```
python scripts/csv_to_npz.py --input_file {your_csvfile}.csv --input_fps 30 --output_name {output_npz_name} --headless
python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 --motion_file {npz_file}.npz  --run_name {} --headless
```
