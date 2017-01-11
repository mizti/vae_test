import numpy as np
import argparse
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
import net.cbr

class Encoder(Chain):
    def __init__(self, in_ch, dim_z):
        super().__init__(
            w = chainer.initializers.Normal(0.02),
            e1 = CBR(in_ch, 32, bn=True, sample='down', activation=F.leaky_relu, dropout=False),
            e2 = CBR(32, 64, bn=True, sample='down', activation=F.leaky_relu, dropout=False),
            e3 = CBR(64, 256, bn=True, sample='down', activation=F.leaky_relu, dropout=False),
            e4 = CBR(256, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False), 
            e5_mu    = L.Convolution2D(512, 512, 4, stride=4),
            e5_sigma = L.Convolution2D(512, 512, 4, stride=4),
            out_mu = L.Linear(512, dim_z),
            out_sigma = L.Linear(512, dim_z)
        )
    def __call__(self, x):
        h = x
        h = F.max_pooling_2d(self.e1(h, test=False), 2)
        h = F.max_pooling_2d(self.e2(h, test=False), 2)
        h = F.max_pooling_2d(self.e3(h, test=False), 2)
        h = F.average_pooling_2d(self.e4(h, test=False), 2)
        mu = self.out_mu(F.sigmoid(e5_mu(h)))
        sigma = self.out_sigma(F.sigmoid(e5_sigma(h)))
        return mu, sigma

class Decoder(Chain):
    def __init__(self, dim_z):
        super().__init__(
            lin=L.Linear(dim_z, 32 * 4 * 4),
            norm0=L.BatchNormalization(32),
            d1 = CBR(32, 512, bn=True, sample='up', activation=F.leaky_relu, dropout=False, ksize=4, stride=2, padding=1),
            d2 = CBR(64, 256, bn=True, sample='up', activation=F.leaky_relu, dropout=False, ksize=4, stride=2, padding=1),
            d3 = CBR(32, 64, bn=True, sample='up', activation=F.leaky_relu, dropout=False, ksize=4, stride=2, padding=1),
            d_r = L.Deconvolution2D(512, 256, 4, stride=2, pad=1),
            d_g = L.Deconvolution2D(512, 256, 4, stride=2, pad=1),
            d_b = L.Deconvolution2D(512, 256, 4, stride=2, pad=1)
        )

    def __call__(self, z):
        h = F.reshape(self.lin(z), (z.data.shape[0], 32, 4, 4))
        h = self.norm0(h)
        h = self.e1(h, test=False)
        h = self.e2(h, test=False)
        h = self.e3(h, test=False)
        r = self.dc_r(h)
        g = self.dc_g(h)
        b = self.dc_b(h)
        return r, g, b

class VAE(Chain):
    def __init__(self, k=512):
        self.k = k
        super().__init__(
            enc = Encoder(k),
            dec = Decoder(k)
        )

    def __call__(self, x, test=False, k=4):
        batch_size = x.data.shape[0]
        w = x.data.shape[2]
        tr, tg, tb = chainer.functions.split_asix(x, 3, 1)
        tr = F.reshape(tr, (batch_size * w * w, ))
        tg = F.reshape(tg, (batch_size * w * w, ))
        tb = F.reshape(tb, (batch_size * w * w, ))

        x = chainer.Variable(x.data.astype('f'))

        z_mu, z_var = self.enc(x, test)
        loss_kl = F.gaussian_kl_divergence(z_mu, z_var) / batch_size / self.k # log of var?
        
        

exit()
parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()

train_data = FontImageDataset(10000, train=True, flatten=False)
test_data = FontImageDataset(10000, train=False, flatten=False)
train_iter = iterators.SerialIterator(train_data, batch_size=200, shuffle=True)
test_iter = iterators.SerialIterator(test_data, batch_size=200, repeat=False, shuffle=False)

model = L.Classifier(CNN())

if args.gpu >= 0:
    chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
    model.to_gpu()  # Copy the model to the GPU

optimizer = optimizers.SGD()
optimizer.setup(model)

#updater = training.StandardUpdater(train_iter, optimizer, device=-1)
updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)

trainer = training.Trainer(updater, (500, 'epoch'), out='result')
print("start running")
#trainer.extend(extensions.Evaluator(test_iter, model))
trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(['epoch', 'main/accuracy', 'validation/main/accuracy']))
#trainer.extend(extensions.PrintReport(['epoch', 'main/accuracy']))
trainer.extend(extensions.ProgressBar())
trainer.run()
print("end running")
