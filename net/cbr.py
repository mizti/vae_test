from __future__ import print_function

import numpy
import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L

# convolution-batchnormalization-(dropout)-relu
class CBR(chainer.Chain):
    def __init__(self, ch0, ch1, bn=True, sample='down', activation=F.relu, dropout=False, ksize=4, stride=2, padding=1):
        self.bn = bn
        self.activation = activation
        self.dropout = dropout
        layers = {}
        w = chainer.initializers.Normal(0.02)
        if sample=='down':
            layers['c'] = L.Convolution2D(ch0, ch1, ksize, stride, padding, initialW=w)
        else:
            layers['c'] = L.Deconvolution2D(ch0, ch1, ksize, stride, pading, initialW=w)
        if bn:
            layers['batchnorm'] = L.BatchNormalization(ch1)
        super(CBR, self).__init__(**layers)
        
    def __call__(self, x, test):
        h = self.c(x)
        if self.bn:
            h = self.batchnorm(h, test=test)
        if self.dropout:
            h = F.dropout(h)
        if not self.activation is None:
            h = self.activation(h)
        return h
