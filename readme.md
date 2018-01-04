## Unsupervised learning by predicting noise

This repo contains a minimal implementation for 'unsupervised learning
predicting noise' paper from FAIR available on [arXiv][paper].

The current pytorch implementation only supports CIFAR-10.


#### Requirements

All the code is written in python3 using pythorch and has been tested on
linux machine. it can be obtained using:

1) Get the code
```
git clone github.com/fair_noise_as_targets
```

2) Create a virtual environment
```
virtualenv noise_as_target
source noise_as_target/bin/activate
```

3) install all dependencies

You will also need to install the pytorch package on the virtual environment
manually. to do so, follow the steps from [pytorch][pytorch]

```
pip3 install -r requirements.txt
```

#### Dataset

Currently, only CIFAR-10 is supported and is available transparently
via the torchvision package.


#### Model

Currently use a Resnet-50 model split into an encoder (all layers but last)
and a decoder (input from the encoder & last FC layer).

#### Training procedure

The current training procedure include a scheme that jointly run both
supervised and unsupervised training. It trains an encoder using the noise
as targets unsupervised training procedure and train a single layer MLP at
regular interval to evaluate the performance of the learnt feature space.


#### Further work

1. move hungarian algorithm computation to GPU (Speed up drastically learning on large batches)
2. support imagenet & custom image datasets.
3. add feature space visualisation notebook (classify samples & run NN clustering)



[pytorch]: http://pytorch.org/
[paper]: https://arxiv.org/abs/1704.05310