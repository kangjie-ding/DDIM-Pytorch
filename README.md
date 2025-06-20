# DDPM && DDIM-Pytorch
In this project, we implemented the training and sampling of DDPM and DDIM with a focus on simplicity and readability. The code is adapted from this existing great [work](https://github.com/xiaohu2015/nngen) that implements some AIGC algorithms, which greatly helped me learn and practice the fundamental algorithms in this field. To validate the effectiveness, we only provided data loading for a few datasets available in torchvision, but of course, you can train the model on your own custom dataset.
# Training
You can easily run the following code to start training.
```sh
python .\train_process.py  
```
# Testing
After obtaining the weights through training, run the following code to test image generation.
```sh
python .\generator.py        
```
# Reference
[DDPM](https://arxiv.org/pdf/2006.11239)

[DDIM](https://arxiv.org/pdf/2010.02502)
