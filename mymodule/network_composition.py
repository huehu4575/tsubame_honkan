#!python3.6.6
import random
import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F

class DeepCNN(chainer.ChainList):

    def __init__(self, n_output):
        super(DeepCNN, self).__init__(
            ConvBlock(128),
            ConvBlock(128, True),
            ConvBlock(256),
            ConvBlock(256, True),
            LinearBlock(),
            LinearBlock(),
            L.Linear(None, n_output)
        )

    def __call__(self, x):
        for f in self:
            x = f(x)
        return x

class SmallDeepCNN(chainer.ChainList):

    def __init__(self, n_output):
        super(SmallDeepCNN, self).__init__(
            ConvBlock(64),
            ConvBlock(64, True),
            ConvBlock(128),
            ConvBlock(128, True),
            LinearBlock(),
            LinearBlock(),
            L.Linear(None, n_output)
        )

    def __call__(self, x):
        for f in self:
            x = f(x)
        return x

class MyNet(chainer.Chain):

    def __init__(self, n_out):
        super(MyNet, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 32, 3, 1, 1)
            self.conv2 = L.Convolution2D(32, 64, 3, 1, 1)
            self.conv3 = L.Convolution2D(64, 128, 3, 1, 1)
            self.fc4 = L.Linear(None, 1000)
            self.fc5 = L.Linear(1000, n_out)

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.fc4(h))
        h = self.fc5(h)
        return h

class ConvBlock(chainer.Chain):

    def __init__(self, n_ch, pool_drop=False):
        w = chainer.initializers.HeNormal()
        super(ConvBlock, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(None, n_ch, 3, 3, 1, nobias=True, initialW=w)
            self.bn = L.BatchNormalization(n_ch)
        self.pool_drop = pool_drop

    def __call__(self, x):
        h = F.relu(self.bn(self.conv(x)))
        if self.pool_drop:
            h = F.max_pooling_2d(h, 2, 2)
            h = F.dropout(h, ratio=0.25)
        return h

class LinearBlock(chainer.Chain):

    def __init__(self, drop=False):
        w = chainer.initializers.HeNormal()
        super(LinearBlock, self).__init__()
        with self.init_scope():
            self.fc = L.Linear(None, 1024, initialW=w)
        self.drop = drop

    def __call__(self, x):
        h = F.relu(self.fc(x))
        if self.drop:
            h = F.dropout(h)
        return h

class AlexNet(chainer.Chain):
    def __init__(self, n_out, train=True):
        super(AlexNet, self).__init__()
        with self.init_scope():
            self.conv1=L.Convolution2D(None, 96, 11, stride=2)
            self.conv2=L.Convolution2D(None, 256, 5, pad=2)
            self.conv3=L.Convolution2D(None, 384, 3, pad=1)
            self.conv4=L.Convolution2D(None, 384, 3, pad=1)
            self.conv5=L.Convolution2D(None, 256, 3, pad=1)
            self.fc6=L.Linear(None, 4096)
            self.fc7=L.Linear(None, 4096)
            self.fc8=L.Linear(None, n_out)

    def __call__(self, x):
        h = F.max_pooling_2d(F.local_response_normalization(F.relu(self.conv1(x))), 3, stride=2)
        h = F.max_pooling_2d(F.local_response_normalization(F.relu(self.conv2(h))), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)
        h = F.dropout(F.relu(self.fc6(h)))
        h = F.dropout(F.relu(self.fc7(h)))
        h = self.fc8(h)

        return h

class LeNet(chainer.Chain):
    def __init__(self, n_out, train=True):
        super(LeNet, self).__init__()
        with self.init_scope():
            self.conv1=L.Convolution2D(None, 6, 5, stride=1)
            self.conv2=L.Convolution2D(None, 16, 5, stride=1)
            self.fc3=L.Linear(None, 120)
            self.fc4=L.Linear(None, 64)
            self.fc5=L.Linear(None, n_out)

    def __call__(self, x):
        h = F.max_pooling_2d(F.local_response_normalization(F.sigmoid(self.conv1(x))), 2, stride=2)
        h = F.max_pooling_2d(F.local_response_normalization(F.sigmoid(self.conv2(h))), 2, stride=2)
        h = F.sigmoid(self.fc3(h))
        h = F.sigmoid(self.fc4(h))
        h = self.fc5(h)

        return h

class VGG16Net(chainer.Chain):
    def __init__(self, n_out, train=True):
        super(VGG16Net, self).__init__()
        with self.init_scope():
            self.conv1=L.Convolution2D(None, 64, 3, stride=1, pad=1)
            self.conv2=L.Convolution2D(None, 64, 3, stride=1, pad=1)

            self.conv3=L.Convolution2D(None, 128, 3, stride=1, pad=1)
            self.conv4=L.Convolution2D(None, 128, 3, stride=1, pad=1)

            self.conv5=L.Convolution2D(None, 256, 3, stride=1, pad=1)
            self.conv6=L.Convolution2D(None, 256, 3, stride=1, pad=1)
            self.conv7=L.Convolution2D(None, 256, 3, stride=1, pad=1)

            self.conv8=L.Convolution2D(None, 512, 3, stride=1, pad=1)
            self.conv9=L.Convolution2D(None, 512, 3, stride=1, pad=1)
            self.conv10=L.Convolution2D(None, 512, 3, stride=1, pad=1)

            self.conv11=L.Convolution2D(None, 512, 3, stride=1, pad=1)
            self.conv12=L.Convolution2D(None, 512, 3, stride=1, pad=1)
            self.conv13=L.Convolution2D(None, 512, 3, stride=1, pad=1)

            self.fc14=L.Linear(None, 4096)
            self.fc15=L.Linear(None, 4096)
            self.fc16=L.Linear(None, n_out)

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(F.local_response_normalization(F.relu(self.conv2(h))), 2, stride=2)

        h = F.relu(self.conv3(h))
        h = F.max_pooling_2d(F.local_response_normalization(F.relu(self.conv4(h))), 2, stride=2)

        h = F.relu(self.conv5(h))
        h = F.relu(self.conv6(h))
        h = F.max_pooling_2d(F.local_response_normalization(F.relu(self.conv7(h))), 2, stride=2)

        h = F.relu(self.conv8(h))
        h = F.relu(self.conv9(h))
        h = F.max_pooling_2d(F.local_response_normalization(F.relu(self.conv10(h))), 2, stride=2)

        h = F.relu(self.conv11(h))
        h = F.relu(self.conv12(h))
        h = F.max_pooling_2d(F.local_response_normalization(F.relu(self.conv13(h))), 2, stride=2)

        h = F.dropout(F.relu(self.fc14(h)))
        h = F.dropout(F.relu(self.fc15(h)))
        h = self.fc16(h)

        return h

class VGG19Net(chainer.Chain):
    def __init__(self, n_out, train=True):
        super(VGG19Net, self).__init__()
        with self.init_scope():
            self.conv1=L.Convolution2D(None, 64, 3, stride=1, pad=1)
            self.conv2=L.Convolution2D(None, 64, 3, stride=1, pad=1)

            self.conv3=L.Convolution2D(None, 128, 3, stride=1, pad=1)
            self.conv4=L.Convolution2D(None, 128, 3, stride=1, pad=1)

            self.conv5=L.Convolution2D(None, 256, 3, stride=1, pad=1)
            self.conv6=L.Convolution2D(None, 256, 3, stride=1, pad=1)
            self.conv7=L.Convolution2D(None, 256, 3, stride=1, pad=1)
            self.conv8=L.Convolution2D(None, 512, 3, stride=1, pad=1)

            self.conv9=L.Convolution2D(None, 512, 3, stride=1, pad=1)
            self.conv10=L.Convolution2D(None, 512, 3, stride=1, pad=1)
            self.conv11=L.Convolution2D(None, 512, 3, stride=1, pad=1)
            self.conv12=L.Convolution2D(None, 512, 3, stride=1, pad=1)

            self.conv13=L.Convolution2D(None, 512, 3, stride=1, pad=1)
            self.conv14=L.Convolution2D(None, 512, 3, stride=1, pad=1)
            self.conv15=L.Convolution2D(None, 512, 3, stride=1, pad=1)
            self.conv16=L.Convolution2D(None, 512, 3, stride=1, pad=1)

            self.fc17=L.Linear(None, 4096)
            self.fc18=L.Linear(None, 4096)
            self.fc19=L.Linear(None, n_out)

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(F.local_response_normalization(F.relu(self.conv2(h))), 2, stride=2)

        h = F.relu(self.conv3(h))
        h = F.max_pooling_2d(F.local_response_normalization(F.relu(self.conv4(h))), 2, stride=2)

        h = F.relu(self.conv5(h))
        h = F.relu(self.conv6(h))
        h = F.relu(self.conv7(h))
        h = F.max_pooling_2d(F.local_response_normalization(F.relu(self.conv8(h))), 2, stride=2)

        h = F.relu(self.conv9(h))
        h = F.relu(self.conv10(h))
        h = F.relu(self.conv11(h))
        h = F.max_pooling_2d(F.local_response_normalization(F.relu(self.conv12(h))), 2, stride=2)

        h = F.relu(self.conv13(h))
        h = F.relu(self.conv14(h))
        h = F.relu(self.conv15(h))
        h = F.max_pooling_2d(F.local_response_normalization(F.relu(self.conv16(h))), 2, stride=2)

        h = F.dropout(F.relu(self.fc17(h)))
        h = F.dropout(F.relu(self.fc18(h)))
        h = self.fc19(h)

        return h
