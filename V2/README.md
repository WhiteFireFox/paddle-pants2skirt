# 实现关键点

&emsp;&emsp;&emsp;&emsp;<font size=4>1、将原图与实例分割图concat后，通过“RGBA”的方式读入(如下图所示)</font><br><br>
&emsp;&emsp;&emsp;&emsp;![](https://ai-studio-static-online.cdn.bcebos.com/bcc459226e094323bdd232ad03a752496b7f7a4e869644a5b69eee7e0f3a8713)<br><br>
&emsp;&emsp;&emsp;&emsp;<font size=4>2、参考CycleGAN引入cyc_loss，G_loss，idt_loss；参考InstaGAN引入bgd_loss(InstaGAN作者将其定义为ctx_loss)；在奇思妙想中新构造了一个seg_loss使得GAN能够学习到实例中的物体的信息实现特定实例的转换。</font><br><br>
&emsp;&emsp;&emsp;&emsp;![](https://ai-studio-static-online.cdn.bcebos.com/480b8382803a45289fb576fe9aaee5b687c5efb0f69d47398d50df72f5568977)

# 实现过程

&emsp;&emsp;&emsp;&emsp;<font size=4>对paddle模型库里面的CycleGAN模型进行微调，对数据处理部分进行改动使得模型能够读入"RGBA"四通道的图片，新增bgd_loss与seg_loss使得模型对实例部分转换更精准高效。</font>

# 项目主体

## 数据处理


```python
!unzip ./data/data52739/dataset.zip -d ./
```


```python
!python ./generate_ccp_dataset.py
```


```python
!mkdir ./dataset/dataset/
!mkdir ./dataset/dataset/trainA/
!mkdir ./dataset/dataset/trainB/
```


```python
import cv2
import os
from PIL import Image
import numpy as np

trainA_seg_path = './dataset/dataset_origin/trainA_seg/'
trainA_path = './dataset/dataset_origin/trainA/'
trainB_seg_path = './dataset/dataset_origin/trainB_seg/'
trainB_path = './dataset/dataset_origin/trainB/'

dir1 = os.listdir(trainA_path)
dir2 = os.listdir(trainB_path)

for i in range(len(dir1)):
    img1 = cv2.imread(trainA_path+dir1[i])
    img1 = img1[:, :, (2, 1, 0)]
    img2 = cv2.imread(trainA_seg_path+dir1[i].split('.')[0]+'_0.png')

    img = np.concatenate((img1, (img2[:, :, 0]*0.21+200)[:, :, np.newaxis]), 2)

    img = Image.fromarray(img.astype(np.uint8), mode="RGBA")
    img.save('./dataset/dataset/trainA/'+dir1[i])

for i in range(len(dir2)):
    img1 = cv2.imread(trainB_path+dir2[i])
    img1 = img1[:, :, (2, 1, 0)]
    img2 = cv2.imread(trainB_seg_path+dir2[i].split('.')[0]+'_0.png')

    img = np.concatenate((img1, (img2[:, :, 0]*0.21+200)[:, :, np.newaxis]), 2)

    img = Image.fromarray(img.astype(np.uint8), mode="RGBA")
    img.save('./dataset/dataset/trainB/'+dir2[i])
```


```python
import os

photo_path1 = './dataset/dataset/trainA'
photo_path2 = './dataset/dataset/trainB'
file_path1 = './dataset/dataset/trainA.txt'
file_path2 = './dataset/dataset/testA.txt'
file_path3 = './dataset/dataset/trainB.txt'
file_path4 = './dataset/dataset/testB.txt'

dir1 = os.listdir(photo_path1)
dir2 = os.listdir(photo_path2)

f1 = open(file_path1, 'w')
f2 = open(file_path2, 'w')
f3 = open(file_path3, 'w')
f4 = open(file_path4, 'w')

for m in range(len(dir1)):
    if m % 100 != 0:
        f1.write('trainA/'+dir1[m]+'\n')
    else:
        f2.write('trainA/'+dir1[m]+'\n')
f1.close()
f2.close()

for n in range(len(dir2)):
    if n % 150 != 0:
        f3.write('trainB/'+dir2[n]+'\n')
    else:
        f4.write('trainB/'+dir2[n]+'\n')
f3.close()
f4.close()
```

## 模型训练


```python
!python instagan/train.py --model_net InstaGAN \
                        --dataset /home/aistudio/dataset/dataset/ \
                        --batch_size 1 \
                        --net_G resnet_9block \
                        --g_base_dim 32 \
                        --net_D basic \
                        --norm_type batch_norm \
                        --epoch 200 \
                        --image_size 286 \
                        --crop_size 256 \
                        --crop_type Random \
                        --output ./output/instagan/
```
