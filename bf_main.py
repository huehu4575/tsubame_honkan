#!python3.6.6
from __future__ import print_function
import random
import time
import numpy as np
import chainer
import argparse
from chainer import iterators
import chainer.links as L
import chainer.functions as F
from chainer import cuda, Function, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
from chainer.training import extensions
from chainer import optimizers
from chainer import training
from chainer.cuda import to_cpu
from functools import partial
from chainercv import transforms
from chainer.datasets import TransformDataset

from mymodule import create_dataset
from mymodule import network_composition
from mymodule import network_composition_01

#############################################################
def reset_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
#    if chainer.cuda.available:
#        chainer.cuda.cupy.random.seed(seed)

def train(network_object, dataset, testdataset, batchsize=128, gpu_id=0, max_epoch=20, postfix='', base_lr=0.01, lr_decay=None):
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=400, help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20, help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu0', '-g', type=int, default=0, help='First GPU ID')
    parser.add_argument('--gpu1', '-G', type=int, default=1, help='Second GPU ID')
    parser.add_argument('--gpu2', type=int, default=2, help='Third GPU ID')
    parser.add_argument('--gpu3', type=int, default=3, help='Fourth GPU ID')
    parser.add_argument('--out', '-o', default='result_parallel', help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='', help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1000, help='Number of units')
    args = parser.parse_args()

    # prepare dataset
    train_size = int(len(dataset) * 1.0)
    train, _ = chainer.datasets.split_dataset_random(Honkan_dataset, train_size, seed=0)
    #train_size = int(len(train_val) * 0.9)
    #train, valid = chainer.datasets.split_dataset_random(train_val, train_size, seed=0)
    test_size = int(len(testdataset) * 0.2)
    test, valid = chainer.datasets.split_dataset_random(Honkan_testdataset, test_size, seed=0)
    # data augement
    train_dataset = TransformDataset(train, partial(transform, train=True))
    valid_dataset = TransformDataset(valid, partial(transform, train=True))
    test_dataset = TransformDataset(test, partial(transform, train=True))

    # 2. Iterator
    #train_iter = iterators.SerialIterator(train, batchsize)
    train_iter = iterators.MultiprocessIterator(train, batchsize)
    #train_iter = iterators.MultiprocessIterator(train, batchsize, n_processes=1)
    #valid_iter = iterators.SerialIterator(valid, batchsize, False, False)
    valid_iter = iterators.MultiprocessIterator(valid, batchsize, False, False)
    #valid_iter = chainer.iterators.MultiprocessIterator(valid, batchsize, repeat=False,shuffle=False, n_processes=1)

    # 3. Model
    net = L.Classifier(network_object)
#    if gpu_id >= 0:
#        cuda.check_cuda_available()
#        chainer.cuda.get_device(gpu_id).use()
#        net.to_gpu(gpu_id)

    # 4. Optimizer
    optimizer = optimizers.MomentumSGD(lr=base_lr).setup(net)
    #optimizer = optimizers.Adam().setup(net)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.0005))

    # 5. Updater
#    updater = training.StandardUpdater(train_iter, optimizer, device=gpu_id)
    updater = training.updaters.ParallelUpdater(train_iter, optimizer, devices={'main': args.gpu0, 'second': args.gpu1, 'third': args.gpu2, 'fourth': args.gpu3 },)

    # 6. Trainer
    trainer = training.Trainer(updater, (max_epoch, 'epoch'), out='{}_BF_Honkan_result_{}'.format(network_object.__class__.__name__, postfix))


    # 7. Trainer extensions
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'))
    #trainer.extend(extensions.Evaluator(valid_iter, net, device=gpu_id), name='val')
    trainer.extend(extensions.Evaluator(valid_iter, net, device=args.gpu0), name='val')
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', 'val/main/loss', 'val/main/accuracy', 'l1/W/data/std', 'elapsed_time']))
    trainer.extend(extensions.PlotReport(['l1/W/data/std'], x_key='epoch', file_name='std.png'))
    trainer.extend(extensions.PlotReport(['main/loss', 'val/main/loss'], x_key='epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/accuracy', 'val/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.ProgressBar())
    # 定期的に状態をシリアライズ（保存）する機能
    trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'))
    trainer.extend(extensions.snapshot_object(net.predictor, filename='model_epoch-{.updater.epoch}'))


    if lr_decay is not None:
        trainer.extend(extensions.ExponentialShift('lr', 0.1), trigger=lr_decay)
    trainer.run()
    del trainer

    # 8. Evaluation
    test_iter = iterators.SerialIterator(test, batchsize, False, False)
    #test_iter = iterators.MultiprocessIterator(test, batchsize, False, False)
    #test_evaluator = extensions.Evaluator(test_iter, net, device=gpu_id)
    test_evaluator = extensions.Evaluator(test_iter, net, device=args.gpu0)
    results = test_evaluator()
    print('Test accuracy:', results['main/accuracy'])

    out_put = '{}_BF_Honkan_result_{}'.format(network_object.__class__.__name__, postfix)
    save_model = out_put + '/model.model'
    save_optimizer = out_put + '/model_optimizer.npz'
    chainer.serializers.save_npz(save_model, net)
    chainer.serializers.save_npz(save_optimizer, optimizer)

    return net

def transform(inputs, train=True):
    img, label = inputs
#    img = img.copy()

    ## Standardization
    #img -= mean[:, None, None]
    #img /= std[:, None, None]

    # Random flip & crop
#    if train:
#        img = transforms.random_crop(img, (1000, 1000))
    return img, label


now = time.ctime()
cnvtime = time.strptime(now)
TimeName = time.strftime("%Y_%m_%d_%H%M", cnvtime)
#######################################################
# set random seed
# 結果保証
reset_seed(0)

#chainer.cuda.set_max_workspace_size(chainer.cuda.get_max_workspace_size())
#print(chainer.cuda.get_max_workspace_size())
chainer.cuda.set_max_workspace_size(512 * 1024 * 1024)
chainer.config.autotune = True

# dataset path
# imageの入っているpathを指定
dir_root = r'/gs/hs0/tga-systemcontrolproject/TRdata_cropped_2/selected_images/'
test_root = r'/gs/hs0/tga-systemcontrolproject/TSdata/selected_images/'
img_root1  = r'bf00'
img_root2  = r'bf01'
img_root3  = r'bf02'
img_root4  = r'bf03'
img_root5  = r'bf04'
img_root6  = r'bf05'
img_root7  = r'bf06'
img_root8  = r'bf07'
img_root9  = r'bf08'
img_root10 = r'bf09'
img_root11 = r'bf10'
img_root12 = r'bf11'
img_root13 = r'bf12'
img_root = [img_root1, img_root2, img_root3, img_root4, img_root5, img_root6, img_root7, img_root8, img_root9, 
            img_root10, img_root11, img_root12, img_root13]

N = 4000; # １クラス当たりN個乱数で，抽出する．
M=70;
Honkan_dataset = create_dataset.create_data_set(dir_root, img_root,[0,1,2,3,4,5,6,7,8,9,10,11,12], N)
print(len(Honkan_dataset))
Honkan_testdataset = create_dataset.create_data_set(test_root,
img_root,[0,1,2,3,4,5,6,7,8,9,10,11,12],M)
print(len(Honkan_testdataset))
#print(Honkan_dataset[100][1])
#model = train(network_composition.LeNet(17), Honkan_dataset, batchsize=5, max_epoch=10, base_lr=0.01, lr_decay=(30, 'epoch'), postfix=TimeName)
#model = train(network_composition.DeepCNN(17), Honkan_dataset, batchsize=5, max_epoch=60, base_lr=0.01, lr_decay=(30, 'epoch'), postfix=TimeName)
#model = train(network_composition.MyNet(17), Honkan_dataset, batchsize=1, max_epoch=100, base_lr=0.01, lr_decay=(30, 'epoch'))
#model = train(network_composition.DeepCNN(35), Honkan_dataset, batchsize=12, max_epoch=60, base_lr=0.01, lr_decay=(30, 'epoch'), postfix=TimeName)
model = train(network_composition.DeepCNN(13),
Honkan_dataset,Honkan_testdataset,batchsize=300,max_epoch=160, base_lr=0.01, lr_decay=(30, 'epoch'), postfix=TimeName)
