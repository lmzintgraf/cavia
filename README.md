## CAVIA (in PyTorch)

Code for "[Fast Context Adaptation via Meta-Learning](https://arxiv.org/abs/1810.03642)" - 
Luisa M Zintgraf, Kyriacos Shiarlis, Vitaly Kurin, Katja Hofmann, Shimon Whiteson
(ICML 2019).

### Regression

See `src/regression/`, 
which includes code for both the sine curve and the CelebA experiments. 
See the file `cavia.py` and `maml.py` and execute them to run experiments.
The settings/hyperparameters can be found and changed in `configs_default.py`.

If you want to use the code for the CelebA dataset, you have to download it 
(`http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html`) and change the path in 
`tasks_celeba.py`.

### Classification

You need the Mini-Imagenet dataset to run these experiments. 
See e.g. `https://github.com/y2l/mini-imagenet-tools` for how to retrieve it.

To run the experiment, execute `cavia.py`.

### Reinforcement Learning

Coming soon.

#

#### Acknowledgements

Special thanks to 
Chelsea Finn, 
Jackie Loong and 
Tristan Deleu for their open-sourced MAML implementations.
This was of great help to us, 
and large parts of our implementation are based on the PyTorch code from:
- Jackie Loong's implementation of MAML, `https://github.com/dragen1860/MAML-Pytorch`
- Tristan Deleu's implementation of MAML-RL, `https://github.com/tristandeleu/pytorch-maml-rl`

#### BibTex

```
@article{zintgraf2018cavia,
  title={Fast Context Adaptation via Meta-Learning},
  author={Zintgraf, Luisa M and Shiarlis, Kyriacos and Kurin, Vitaly and Hofmann, Katja and Whiteson, Shimon},
  conference={Thirty-sixth International Conference on Machine Learning (ICML 2019)},
  year={2019}
}
```