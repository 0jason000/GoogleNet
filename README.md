# GoogleNet

## GoogleNet描述

GoogleNet是2014年提出的22层深度网络，在2014年ImageNet大型视觉识别挑战赛（ILSVRC14）中获得第一名。  GoogleNet，也称Inception v1，比ZFNet（2013年获奖者）和AlexNet（2012年获奖者）改进明显，与VGGNet相比，错误率相对较低。  深度学习网络包含的参数更多，更容易过拟合。网络规模变大也会增加使用计算资源。为了解决这些问题，GoogleNet采用1*1卷积核来降维，从而进一步减少计算量。在网络末端使用全局平均池化，而不是使用全连接的层。  inception模块为相同的输入设置不同大小的卷积，并堆叠所有输出。

[论文](https://arxiv.org/abs/1409.4842)：Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich."Going deeper with convolutions."*Proceedings of the IEEE conference on computer vision and pattern recognition*.2015.

## 模型架构

GoogleNet由多个inception模块串联起来，可以更加深入。  降维的inception模块一般包括**1×1卷积**、**3×3卷积**、**5×5卷积**和**3×3最大池化**，同时完成前一次的输入，并在输出处再次堆叠在一起。

## 训练过程

### 训练


- GPU处理器环境运行

  ```bash
  export CUDA_VISIBLE_DEVICES=0
  python train.py --model googlenet --data_url ./dataset/imagenet > train.log 2>&1 &
  ```

  上述python命令将在后台运行，您可以通过train.log文件查看结果。

  训练结束后，您可在默认`./ckpt_0/`脚本文件夹下找到检查点文件。


## 评估过程

### 评估

- 在GPU处理器环境运行时评估CIFAR-10数据集

  在运行以下命令之前，请检查用于评估的检查点路径。请将检查点路径设置为绝对全路径，例如“username/googlenet/train/ckpt_0/train_googlenet_cifar10-125_390.ckpt”。

  ```bash
  python train.py --model googlenet --data_url ./dataset/imagenet --checkpoint_path=[CHECKPOINT_PATH] > eval.log 2>&1 &
  ```

  上述python命令将在后台运行，您可以通过eval.log文件查看结果。测试数据集的准确性如下：

  ```bash
  # grep "accuracy:" eval.log
  accuracy:{'acc':0.930}
  ```


