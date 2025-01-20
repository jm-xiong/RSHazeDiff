# RSHazeDiff: A Unified Fourier-aware Diffusion Model for Remote Sensing lmage Dehazing (TITS2024)

[Jiamei Xiong](https://scholar.google.com/citations?user=-epzM98AAAAJ&hl=en), [Xuefeng Yan](https://scholar.google.com/citations?hl=en&user=C_sZsC0AAAAJ&view_op=list_works&sortby=pubdate), [Yongzhen Wang](https://scholar.google.com/citations?hl=en&user=fLeZQYkAAAAJ), Wei Zhao, [Xiao-Ping Zhang](https://scholar.google.ca/citations?hl=en&user=1fzb_z8AAAAJ&view_op=list_works&sortby=pubdate), [Mingqiang Wei](https://scholar.google.com/citations?user=TdrJj8MAAAAJ&hl=en&oi=ao)

[![paper](https://img.shields.io/badge/paper-TITS2024-blue)](https://ieeexplore.ieee.org/document/10747754)
[![arXiv](https://img.shields.io/badge/arXiv-2405.09083-red)](https://arxiv.org/abs/2405.09083)

#### News
- **Dec 20, 2023:** This repo is released. 
- **May 15, 2024:** Arxiv paper is available.
- **Nov 8, 2024:** 😊 Paper is accepted by IEEE TITS2024. 
- **Jan 17, 2025:** 🔈The code is available now, enjoy yourself!
- **Jan 20, 2025:** Updated README file with detailed instruciton.

<hr />

> **Abstract:** *Haze severely degrades the visual quality of remote sensing images and hampers the performance of road extraction, vehicle detection, and traffic flow monitoring. The emerging denoising diffusion probabilistic model (DDPM) exhibits the significant potential for dense haze removal with its strong generation ability. Since remote sensing images contain extensive small-scale texture structures, it is important to effectively restore image details from hazy images. However, current wisdom of DDPM fails to preserve image details and color fidelity well, limiting its dehazing capacity for remote sensing images. In this paper, we propose a novel unified Fourier-aware diffusion model for remote sensing image dehazing, termed RSHazeDiff. From a new perspective, RSHazeDiff explores the conditional DDPM to improve image quality in dense hazy scenarios, and it makes three key contributions. First, RSHazeDiff refines the training phase of diffusion process by performing noise estimation and reconstruction constraints in a coarse-to-fine fashion. Thus, it remedies the unpleasing results caused by the simple noise estimation constraint in DDPM. Second, by taking the frequency information as important prior knowledge during iterative sampling steps, RSHazeDiff can preserve more texture details and color fidelity in dehazed images. Third, we design a global compensated learning module to utilize the Fourier transform to capture the global dependency features of input images, which can effectively mitigate the effects of boundary artifacts when processing fixed-size patches. Experiments on both synthetic and real-world benchmarks validate the favorable performance of RSHazeDiff over state-of-the-art methods.* 
<hr />

## Network Architecture
<img src = "https://imgur.la/images/2025/01/20/Overview.jpg"> 

⭐If this work is helpful for you, please help star this repo. Thanks!🤗

## Getting Started
### Environment
Clone this repo:

```
git clone https://github.com/jm-xiong/RSHazeDiff.git
cd RSHazeDiff/
```

Create a new conda environment and install dependencies:

```
conda create -n rshazediff python=3.7
conda activate rshazediff
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
```

### Prepare Datasets
You can download the datasets [LHID & DHID](https://pan.baidu.com/s/13aW-khZZcLF3_1ax4H8GXQ?pwd=QW67) (password: QW67) and [RICE](https://drive.google.com/file/d/1CricZtIj28BGFvkD_x-W8fSexPiDtgHk/view). Note that ERICE dataset is expanded on RICE1 by cutting the images to the size of 256 × 256 pixels without overlapping. Make sure the file structure is consistent with the following:

```python
├── ERICE
│   ├── Test
│   │   ├── GT
│   │   └── Haze
│   └── Train
│       ├── GT
│       └── Haze
└── HazyRemoteSensingDatasets
    ├── DHID
    │   ├── TestingSet
    │   │   └── Test
    │   │       ├── GT
    │   │       └── Haze
    │   └── TrainingSet
    │       ├── GT
    │       └── Haze
    └── LHID
        ├── TestingSet
        │   └── Merge
        │       ├── GT
        │       └── Haze
        └── TrainingSet
            ├── GT
            └── Haze
```


## Citation
Please cite us if our work is useful for your research.

    @article{xiong2025rshazediff,
        title={RSHazeDiff: A Unified Fourier-aware Diffusion Model for Remote Sensing lmage Dehazing}, 
        author={Xiong, Jiamei and Yan, Xuefeng and Wang, Yongzhen and Wei Zhao and Xiao-Ping Zhang and Wei, Mingqiang},
        journal={IEEE Transactions on Intelligent Transportation Systems},
        volume={26},
        issue={1},
        pages={1055-1070},
        year={2024},
        doi={10.1109/TITS.2024.3487972},
        publisher={IEEE}
    }

## Acknowledgement

This code is based on [WeatherDiffusion](https://github.com/IGITUGraz/WeatherDiffusion). Thanks for their awesome work.

## Contact

If you have any questions, feel free to approach me at jmxiong@nuaa.edu.cn
