# SAC

This is a PyTorch implementation of [Soft Actor Critic](https://arxiv.org/abs/1801.01290), including [Impala Architecture](https://arxiv.org/abs/1802.01561).
  
 # Running the code
 
**training a model.**

```shell script
$ python train.py
```

**testing a model.**

```shell script
$ python test.py PATH
```

Where `PATH` is the path to your model.

**show the logs.**

```shell script
$ tensorboard --logdir LOG --port 6006
```

Where `LOG` is the path of your logs, And display on `localhost:6006/`

# To-do list

Here's my checklist:

  * [x] Implement SAC Algorithm
  * [x] Implement [Temperature](https://arxiv.org/abs/1812.05905) update
  * [x] Implement IMPALA Architecture
