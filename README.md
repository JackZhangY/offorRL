an offline RL repository in progress...


### Done & Planning

- [x] [IQL](https://arxiv.org/abs/2110.06169) 
- [ ] [CQL](https://arxiv.org/abs/2006.04779) 

### Quick Start

We build this project based on the RL framework, [rlkit](https://github.com/rail-berkeley/rlkit), and the offline Mujoco 
benchmark [d4rl](https://github.com/Farama-Foundation/D4RL). And we use [hydra-core](https://hydra.cc) to manage our 
experimental configurations, so make sure of the installations of 'd4rl' before starting, and the installation of 'hydra'
can be done as follows:

```bash
pip install hydra-core --upgrade
pip install hydra-joblib-launcher --upgrade
```
As for 'rlkit', you can install it within our project:
```bash
cd offorRL/d4rl
pip install -e.
```
Then you can begin your offline RL experiments as follows (take 'IQL' as an example):
```bash
python main_off.py --multirun trainer=iql env.name='hopper-medium-v2' 
``` 


