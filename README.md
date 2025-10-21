# UNet Image Segmentation

## Overview
This project implements the UNet architecture using Pytorch for image segmentation. It is trained and evaluated on the Oxford Pets dataset, segmenting pixels into background and foreground regions.

## Features
- UNet architecture with four downsample and four upsample blocks
- Real time training loss and accuracy tracking and pixel wise accuracy
- Supports MPS acceleration

## Project Structure
```bash
image-segmentation-unet/
│
├── .vscode/ # Editor settings
├── data/ # Dataset directory (Oxford Pets)
├── notebooks/ # Jupyter notebooks for experimentation
│ └── sandbox.ipynb
├── src/ # Source code
│ ├── init.py
│ ├── data.py # Dataset loading and preprocessing
│ ├── model.py # U-Net model architecture
│ ├── train.py # Training and validation loops
│ └── utils.py # Helper functions (metrics, transforms)
│
├── tests/ # Unit tests
│ └── test_utils.py
│
├── venv/ # Virtual environment (not tracked in git)
├── main.py # Entry point to train and evaluate model
├── requirements.txt # Dependencies
└── README.md # Project documentation
```
## Set up instructions

### 1. Clone and navigate
```bash
git clone https://github.com/<your-username>/image-segmentation-unet.git
cd image-segmentation-unet
```
### 2. Download and save Oxford Pets dataset
https://www.robots.ox.ac.uk/~vgg/data/pets/
-> save into a directory called data

### 3. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate    # Mac / Linux
venv\Scripts\activate       # Windows
```
### 4. Install dependencies
```bash
pip install -r requirements.txt
```
### 5. Run training
```bash
python3 main.py
```

## Author
Benjamin Wang
Dartmouth College --- Math and Computer Science
benjaminwang2025@gmail.com | benjamin.wang.29@dartmouth.edu

## License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Author's Notes
This is my second independently built ML project this year, and it was significantly more challenging than the simple two-layer network I built over the summer. I started by reading several articles about convolutional neural networks, and I gained a solid understanding of how to correctly apply padding, strides, kernel sizes, and transpose convolutions — specifically thanks to an insightful paper I found on [arXiv](https://arxiv.org/pdf/1603.07285).

This project also introduced me to new PyTorch concepts, in particular, working with custom Dataset and DataLoader classes. I also experimented with MPS acceleration and managing device placement for both the model and data.

My biggest takeaway from building this U-Net was the importance of data preprocessing. Since the U-Net architecture requires concatenating skip tensors during the upsampling blocks, I had to ensure that the input data dimensions were perfectly aligned so that the upsampled outputs matched the skip tensor dimensions.

For future projects, I plan to explore areas beyond computer vision — possibly reinforcement learning (RL) or natural language processing (NLP) — to broaden my exposure to different areas of applied machine learning.


