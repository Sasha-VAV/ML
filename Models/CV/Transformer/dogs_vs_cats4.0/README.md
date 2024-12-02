# Transformer neural network, that choose between dog and cat
***
# Train it to use!
***
## How to launch it?
- run command
```shell
poetry install
```
- To configure network and execute it check main.py,
everything is described in main.py
***
## Current stats:
- Architecture: Transformer
- Train set size: 24000 images
- Test set size: 1000 images
- Max achieved **accuracy**: 95.6
***
## References:
- Based on https://arxiv.org/abs/2010.11929
- Model is based on the previous version you can check it at
ML\Models\CV\CNN\AlexNex\dogs_vs_cats
***
## Max accuracy research
- default LeNet: 74.1
- default AlexNet: 93.9
- AlexNet + data augmentation: 95.6
