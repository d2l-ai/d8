# MXNet

```{.python .input  n=1}
#@hide
%matplotlib inline
%load_ext autoreload
%autoreload 2

from IPython import display
display.set_matplotlib_formats('svg')
```

```{.python .input  n=2}
import mxnet as mx
from mxnet.gluon.data.vision import transforms
import time

def get_dataloader(ds, batch_size, is_train):
    jitter_param = 0.4
    lighting_param = 0.1
    num_workers = 16

    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomFlipLeftRight(),
        transforms.RandomColorJitter(brightness=jitter_param, contrast=jitter_param,
                                     saturation=jitter_param),
        transforms.RandomLighting(lighting_param),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    return mx.gluon.data.DataLoader(
        ds.to_mxnet().transform_first(transform_train if is_train else transform_test),
        batch_size=batch_size, shuffle=True, num_workers=num_workers, last_batch='keep')

def get_model(nclasses):
    #model_name = 'ResNet34_v2'
    finetune_net = mx.gluon.model_zoo.vision.resnet50_v2(pretrained=True)
    finetune_net.output = mx.gluon.nn.Dense(nclasses)
    finetune_net.output.initialize(mx.init.Xavier())
    return finetune_net

def test_acc(net, val_data, ctx):
    metric = mx.metric.Accuracy()
    for i, batch in enumerate(val_data):
        data = mx.gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = mx.gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        outputs = [net(X) for X in data]
        metric.update(label, outputs)
    return metric.get()

def train(ds, hp):
    print(f'{ds.summary()}\n{hp}')

    start_tic = time.time()
    ds, test = ds.split(0.8)
    train, valid = ds.split(0.8)
    train_data = get_dataloader(train, hp.batch_size, is_train=True)
    val_data = get_dataloader(valid, hp.batch_size, is_train=False)
    test_data = get_dataloader(test, hp.batch_size, is_train=False)

    ctx = [mx.gpu(i) for i in range(hp.num_gpus)]

    metric = mx.metric.Accuracy()
    L = mx.gluon.loss.SoftmaxCrossEntropyLoss()

    model = get_model(len(ds.classes))
    model.collect_params().reset_ctx(ctx)
    trainer = mx.gluon.Trainer(model.collect_params(), 'sgd', {
                            'learning_rate': hp.lr, 'momentum': hp.momentum, 'wd': hp.wd})

    lr_counter = 0
    num_batch = len(train_data)

    smooth_val_acc = 0
    for epoch in range(hp.epochs):
        if epoch == hp.lr_steps[lr_counter]:
            trainer.set_learning_rate(trainer.learning_rate*hp.lr_factor)
            lr_counter += 1

        tic = time.time()
        train_loss = 0
        metric.reset()

        for i, batch in enumerate(train_data):
            data = mx.gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
            label = mx.gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
            with mx.autograd.record():
                outputs = [model(X) for X in data]
                loss = [L(yhat, y) for yhat, y in zip(outputs, label)]
            for l in loss:
                l.backward()

            trainer.step(hp.batch_size)
            train_loss += sum([l.mean().asscalar() for l in loss]) / len(loss)

            metric.update(label, outputs)

        _, train_acc = metric.get()
        train_loss /= num_batch

        _, val_acc = test_acc(model, val_data, ctx)

        if epoch == 0:
            smooth_val_acc = val_acc
        else:
            smooth_val_acc = 0.5 * smooth_val_acc + 0.5 * val_acc

        print('[Epoch %d] Train-acc: %.3f, loss: %.3f | Val-acc: %.3f, %.3f | time: %.1f' %
             (epoch, train_acc, train_loss, val_acc, smooth_val_acc, time.time() - tic))

        if epoch > 5 and val_acc < smooth_val_acc + hp.early_stop:
            break

    return {'train_acc':train_acc, 'valid_acc':test_acc(model, test_data, ctx)[1], 'time':time.time() - start_tic}

```

```{.python .input  n=3}
import dataclasses
from typing import Tuple

@dataclasses.dataclass(unsafe_hash=True)
class HP:
    epochs: int = 40
    lr: float = 0.001
    momentum: int = 0.9
    wd: float = 0.0001
    batch_size: int = 64
    lr_factor: float = 0.75
    lr_steps: Tuple[int] = (10, 20, 30,40)
    early_stop: float = 0.001
    num_gpus: int = 4
```

```{.python .input  n=4}
import pathlib
import pickle
from d8.image_classification import Dataset

result_file = pathlib.Path.home()/'.d8/image_classification_mxnet.pkl'

def search(hp, result_file=result_file):
    if result_file.exists():
        results = pickle.load(result_file.open('rb'))
    else:
        results = {}
    if hp not in results:
        results[hp] = {}
    for i, name in enumerate(Dataset.list()):
        if name in results[hp]:
            print(f'skip {name} as found in previous results')
            continue
        print(f'[{i}/{len(Dataset.list())}] start to train dataset "{name}"')
        ds = Dataset.get(name)
        res = train(ds, hp)
        print(res)
        results[hp][name] = res
        pickle.dump(results, result_file.open('wb'))

hp = HP()
search(hp)
```

```{.python .input}
import pandas as pd
pd.set_option('precision', 2)

results = pickle.load(result_file.open('rb'))
names = Dataset.list()
best_results = {}
for name in names:
    for val in results.values():
        if name in val:
            if (name not in best_results or
                best_results[name]['valid_acc'] < val[name]['valid_acc']):
                best_results[name] = val[name]
res_df = pd.DataFrame([best_results[name] for name in names], index=names)
summary = Dataset.summary_all(quick=True)
pd.merge(summary, res_df, left_index=True, right_index=True)
```
