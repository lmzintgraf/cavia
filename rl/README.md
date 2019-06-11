## CAVIA - Reinforcement Learning

This code is an extended version of Tristan Deleu's PyTorch MAML implementation: 
`https://github.com/tristandeleu/pytorch-maml-rl`.

##

### Prerequisites

For the MuJoCo experiments you need [`mujoco-py`](https://github.com/openai/mujoco-py) 
and [OpenAI gym](https://github.com/openai/gym).

### Running experiments

To run an experiment on the 2D navigation, use the following command:

```
python3 main.py --env-name 2DNavigation-v0 --fast-lr 1.0 --phi-size 5 0  --output-folder results
```
