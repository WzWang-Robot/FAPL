# SAN-FAPL
This repository contains source codes for our paper: "Feedback-efficient Active Preference Learning for Socially Aware Robot Navigation" in IROS-2022.
For more details, please refer to [our project website](https://sites.google.com/view/san-fapl) and [arXiv preprint](https://arxiv.org/abs/). 
For experiment demonstrations, please refer to the [youtube video](https://youtu.be/).


## Abstract
Socially aware robot navigation, where a robot is required to optimize its trajectory to maintain comfortable and compliant spatial interactions with humans in addition to reaching its goal without collisions, is a fundamental yet challenging task in the context of human-robot interaction. While existing learning-based methods have achieved better performance than the preceding model-based ones, they still have drawbacks: reinforcement learning depends on the handcrafted reward that is unlikely to effectively quantify broad social compliance, and can lead to reward exploitation problems; meanwhile, inverse reinforcement learning suffers from the need for expensive human demonstrations. In this paper, we propose a feedback-efficient active preference learning approach, FAPL, that distills human comfort and expectation into a reward model to guide the robot agent to explore latent aspects of social compliance. We further introduce hybrid experience learning to improve the efficiency of human feedback and samples, and evaluate benefits of robot behaviors learned from FAPL through extensive simulation experiments and a user study (N=10) employing a physical robot to navigate with human subjects in real-world scenarios.


## Overview Architecture for FAPL
<img src="/figures/architecture.pdf" width="450" />
<img src="/figures/environment.jpg" width="300" />


### Set Up
1. Install the required python package
```
pip install -r requirements.txt
```

2. Install [Human-in-Loop RL Environment](https://github.com/rll-research/BPref) 
```
git clone https://github.com/rll-research/BPref.git
pip install -e .
```

3. Install [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2) library

4. Install Environment and Navigation into pip
```
pip install -e .
```


### Run the code
1. Train a policy with preference learning. 
```
python train_FAPL.py 
```

2. Test policies.
```
python test_FAPL.py --vis 
```

3. Plot training curve.
```
python plot.py
```

4. Demonstration_api.
```
python demonstation_api.py --vis 
```

(We only tested our code in Ubuntu 18.04 with Python 3.6.)

## Learning Curve
Learning curves of FAPL in 360 degrees FoV environment with 10 humans.

<img src="/figures/curve_sr.png" width="450" />
<img src="/figures/curve_df.png" width="450" />

## Citation
If you find the code or the paper useful for your research, please cite our paper:
```
@inproceedings{FAPL,
  title={FAPL},
  author={Ruqii Wang, Weizheng Wang, Professor},
  booktitle={International Conference on Robotics and Automation (IROS)},
  year={2022}
}
```

## Acknowledgement

Other contributors:  
[Weizheng Wang](https://github.com/WzWang-Robot/FAPL)  

Part of the code is based on the following repositories:  

[1] C. Chen, Y. Liu, S. Kreiss, and A. Alahi, “Crowd-robot interaction: Crowd-aware robot navigation with attention-based deep reinforcement learning,” in International Conference on Robotics and Automation (ICRA), 2019, pp. 6015–6022.
(Github: https://github.com/vita-epfl/CrowdNav)

[2] A,B
(Github: https://github.com/Shuijing725/CrowdNav_DSRNN)

[3] K. Lee, L. M. Smith, and P. Abbeel, “Pebble: Feedback-efficient interactive reinforcement learning via relabeling experience and un-supervised pre-training,” in International Conference on Machine
Learning. PMLR, 2021, pp. 6152–6163
(Github: https://github.com/rll-research/B_Pref)




