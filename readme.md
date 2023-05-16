本项目reid采用**表征学习和度量学习**结合的方式进行训练

是在**reid-strong-baseline**基础上实现

度量学习采用三元组损失函数

数据集：mark1501(将数据集mark1501放在data文件夹下)

baseline网络：支持Resnet系列，例如resnet18、resnet34、rensnet_ibn等

# 环境说明

torch 1.7

torchvision 0.8.1

numpy:1.18.5

matplotlib:3.2.2

opencv-python:4.1.2

pytorch-ignite:0.4.11



# Reid训练

```shell
python tools/train.py --model_name resnet50_ibn_a --model_path weights/ReID_resnet50_ibn_a.pth --IMS_PER_BATCH 8 --TEST_IMS_PER_BATCH 4 --MAX_EPOCHS 120
```

**model_name:**可支持的baseline网络

​						 支持：resnet18,resnet34,resnet50,resnet101,resnet50_ibn_a

**model_path:**预权重路径；

**IMS_PER_BATCH:**训练时batch size

**TEST_IMS_PER_BATCH：**测试时batch size

**MAX_EPOCHS:**训练的epochs数



接着会出现下面的内容：

```shell
=> Market1501 loaded
Dataset statistics:
  ----------------------------------------
  subset   | # ids | # images | # cameras
  ----------------------------------------
  train    |   751 |    12936 |         6
  query    |   750 |     3368 |         6
  gallery  |   751 |    15913 |         6
  ----------------------------------------
  
2023-05-15 14:30:55.603 | INFO     | engine.trainer:log_training_loss:119 - Epoch[1] Iteration[227/1484] Loss: 6.767, Acc: 0.000, Base Lr: 3.82e-05
2023-05-15 14:30:55.774 | INFO     | engine.trainer:log_training_loss:119 - Epoch[1] Iteration[228/1484] Loss: 6.761, Acc: 0.000, Base Lr: 3.82e-05
2023-05-15 14:30:55.946 | INFO     | engine.trainer:log_training_loss:119 - Epoch[1] Iteration[229/1484] Loss: 6.757, Acc: 0.000, Base Lr: 3.82e-05
2023-05-15 14:30:56.134 | INFO     | engine.trainer:log_training_loss:119 - Epoch[1] Iteration[230/1484] Loss: 6.760, Acc: 0.000, Base Lr: 3.82e-05
2023-05-15 14:30:56.305 | INFO     | engine.trainer:log_training_loss:119 - Epoch[1] Iteration[231/1484] Loss: 6.764, Acc: 0.000, Base Lr: 3.82e-05

```

每个epoch训练完成后会测试一次mAP：

我这里第一个epoch的mAP达到75.1%，Rank-1:91.7%, Rank-5:97.2%, Rank-10:98.2%。

测试完成后会在log文件下保存一个pth权重，名称为mAPxx.pth，也是用该权重进行测试。

```shell
2023-05-15 14:35:59.753 | INFO     | engine.trainer:print_times:128 - Epoch 1 done. Time per batch: 261.820[s] Speed: 45.4[samples/s]
2023-05-15 14:35:59.755 | INFO     | engine.trainer:print_times:129 - ----------
The test feature is normalized
2023-05-15 14:39:51.025 | INFO     | engine.trainer:log_validation_results:137 - Validation Results - Epoch: 1
2023-05-15 14:39:51.048 | INFO     | engine.trainer:log_validation_results:140 - mAP:75.1%
2023-05-15 14:39:51.051 | INFO     | engine.trainer:log_validation_results:142 - CMC curve, Rank-1  :91.7%
2023-05-15 14:39:51.051 | INFO     | engine.trainer:log_validation_results:142 - CMC curve, Rank-5  :97.2%
2023-05-15 14:39:51.052 | INFO     | engine.trainer:log_validation_results:142 - CMC curve, Rank-10 :98.2%

```

# 测试

```shell
python tools/test.py --TEST_IMS_PER_BATCH 4 --model_name [your model name] --model_path [your weight path]
```

可以进行mAP,Rank的测试

------

# Reid相关资料学习链接

数据集代码详解：https://blog.csdn.net/z240626191s/article/details/130371383?spm=1001.2014.3001.5501

Reid损失函数理论讲解：https://blog.csdn.net/z240626191s/article/details/130405664?spm=1001.2014.3001.5501

Reid度量学习Triplet loss代码讲解：https://blog.csdn.net/z240626191s/article/details/130490628?spm=1001.2014.3001.5501

yolov5 reid项目(支持跨视频检索)：https://blog.csdn.net/z240626191s/article/details/129221510?spm=1001.2014.3001.5501

yolov3 reid项目(支持跨视频检索)：https://blog.csdn.net/z240626191s/article/details/123004326?spm=1001.2014.3001.5502

**预权重链接：**

链接：https://pan.baidu.com/s/10dAj75wRiEZ7vuK8bU4GOg 
提取码：yypn



如果项目对你有用，麻烦点个Star

# 后期计划更新

​					1.引入知识蒸馏训练

​					2.加入YOLOX

