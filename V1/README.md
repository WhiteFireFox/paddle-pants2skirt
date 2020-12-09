## **实现方式**

&emsp;&emsp;&emsp;&emsp;<font size=5>基于ace2p模型+CycleGan</font>

# **项目主体**

## **简介**

&emsp;&emsp;&emsp;&emsp;<font size=5>参考了instagan的思想并进行改动，在原图的基础上，利用语义分割分割出来裤子和裙子，并与原图进行融合，增强裤子和裙子那一部分的特征信息，然后利用CycleGan对裤子和裙子进行生成迁移。</font><br><br>
&emsp;&emsp;&emsp;&emsp;![](https://ai-studio-static-online.cdn.bcebos.com/936716c67d204bf8af7c4e26bdab2005613383f60c564468a43ce24e5693fad0)

## **解压数据集**


```python
!unzip data/data51854/dataset.zip -d /home/aistudio/
```

## **获取ace2p模型**

&emsp;&emsp;&emsp;&emsp;![](https://ai-studio-static-online.cdn.bcebos.com/f6b2d54aabe14da28d05d50a759e69384aae2d05ff3d405b973e8d958ea5d7f3)


```python
!pip install paddlehub==1.6.0
#ace2p 模型
!hub install ace2p==1.1.0
```

## **对图像进行处理**

&emsp;&emsp;&emsp;&emsp;![](https://ai-studio-static-online.cdn.bcebos.com/3f9a8024a1394efb9d8a7c7d2931668418ad2f604f664a7bae106dcfc552652a)


```python
!rm -rf dataset/trainA_seg
!rm -rf dataset/trainB_seg
!rm -rf dataset/trainA_seg_orgin
!rm -rf dataset/trainB_seg_orgin
```


```python
!mkdir dataset/trainA_seg
!mkdir dataset/trainB_seg
!mkdir dataset/trainA_seg_orgin
!mkdir dataset/trainB_seg_orgin
```


```python
!python origin2seg.py
```


```python
!rm -rf dataset/trainA
!rm -rf dataset/trainB
!mkdir dataset/trainA
!mkdir dataset/trainB
!python GetDataset1.py
```

## **CycleGan-1**

&emsp;&emsp;&emsp;&emsp;![](https://ai-studio-static-online.cdn.bcebos.com/d978a7e6a6c349a9892a122f5232a8ac4b3ee66c089540b982e0d183a52f722f)


```python
!python gan/train.py --model_net CycleGAN \
                        --dataset /home/aistudio/dataset/ \
                        --batch_size 1 \
                        --net_G unet_256 \
                        --g_base_dim 32 \
                        --net_D basic \
                        --norm_type batch_norm \
                        --epoch 150 \
                        --image_size 550 \
                        --crop_size 512 \
                        --crop_type Random \
                        --output ./output/cyclegan1/
```

## **准备下一份的数据集**


```python
!mkdir dataset_last
!rm -rf dataset_last/*
!python GetDataTxt1.py
```


```python
!python gan/infer.py --init_model output/cyclegan1/checkpoints/8/ \
                        --dataset_dir /home/aistudio/ \
                        --image_size 512 \
                        --n_samples 1 \
                        --crop_size 512 \
                        --input_style B \
                        --test_list /home/aistudio/dataset_last/originB.txt \
                        --model_net CycleGAN \
                        --net_G unet_256 \
                        --g_base_dims 32 \
                        --output ./dataset_last/originB/
```


```python
import os

path = './dataset_last/originB/'

dir1 = os.listdir(path)

for i in dir1:
    if i.split('_')[0] != 'fake':
        os.remove(path+i)
    else:
        old_name = path+i
        new_name = path+i.split('_')[1]
        os.rename(old_name, new_name)
```


```python
!python GetDataTxt2.py
```


```python
!python gan/infer.py --init_model output/cyclegan1/checkpoints/8/ \
                        --dataset_dir /home/aistudio/dataset_last/ \
                        --image_size 512 \
                        --n_samples 1 \
                        --crop_size 512 \
                        --input_style A \
                        --test_list /home/aistudio/dataset_last/originA.txt \
                        --model_net CycleGAN \
                        --net_G unet_256 \
                        --g_base_dims 32 \
                        --output ./dataset_last/originA/
```


```python
import os

path = './dataset_last/originA/'

dir1 = os.listdir(path)

for i in dir1:
    if i.split('_')[0] != 'fake':
        os.remove(path+i)
    else:
        old_name = path+i
        new_name = path+i.split('_')[1]
        os.rename(old_name, new_name)
```


```python
!rm -rf ./dataset/trainB_mask/
!mkdir ./dataset/trainB_mask/
!python GetDataset2.py
!rm -rf dataset_last/trainA
!rm -rf dataset_last/trainB
!cp -r dataset/trainB_mask dataset_last/trainA
!cp -r dataset_last/originA dataset_last/trainB
!python GetDataTxt3.py
```

## **CycleGan-2**

&emsp;&emsp;&emsp;&emsp;![](https://ai-studio-static-online.cdn.bcebos.com/936716c67d204bf8af7c4e26bdab2005613383f60c564468a43ce24e5693fad0)


```python
 !python gan/train.py --model_net CycleGAN \
                        --dataset /home/aistudio/dataset_last/ \
                        --batch_size 1 \
                        --net_G resnet_9block \
                        --g_base_dim 32 \
                        --net_D basic \
                        --norm_type batch_norm \
                        --epoch 60 \
                        --image_size 512 \
                        --crop_size 256 \
                        --crop_type Random \
                        --output ./output/cyclegan5/     
```

# **小裙子“加工厂”**

&emsp;&emsp;&emsp;&emsp;<font size=4>解压并获取网络参数</font>


```python
!unzip data/data52282/net.zip
!mv home/aistudio/output /home/aistudio/
!rm -rf home/
```


```python
import cv2
import paddlehub as hub
import matplotlib.pyplot as plt 
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#ace2p 模型
module = hub.Module(name="ace2p")
test_img_path = "0348.jpg"
# 预测前展示
img = cv2.imread(test_img_path)

image=cv2.imread(test_img_path)

results = module.segmentation(images = [image], use_gpu=True, output_dir = './',
                                visualization = True)
old_name = './'+results[0]['path'][:-3]+'png'
new_name = './'+test_img_path.split('.')[0]+'.jpg'
os.rename(old_name, new_name)
img = cv2.imread(new_name)
for m in range(img.shape[0]):
    for n in range(img.shape[1]):
        if img[m, n][0] != 0 or img[m, n][1] != 0 or img[m, n][2] != 192:
            img[m, n] = np.array([0, 0, 0])

img = cv2.addWeighted(img, 0.9, image, 0.1, 0)
cv2.imwrite('./'+test_img_path.split('.')[0]+'.jpg', img)

f1 = open('test.txt','w')
f1.write(test_img_path)
f1.close()

f2 = open('test_last.txt','w')
f2.write('fake_'+test_img_path)
f2.close()
```


```python
!python gan/infer.py --init_model output/cyclegan1/checkpoints/9/ \
                        --dataset_dir /home/aistudio/ \
                        --image_size 512 \
                        --n_samples 1 \
                        --crop_size 512 \
                        --input_style A \
                        --test_list /home/aistudio/test.txt \
                        --model_net CycleGAN \
                        --net_G unet_256 \
                        --g_base_dims 32 \
                        --output ./
```


```python
import cv2

img1 = cv2.imread(test_img_path)
img2 = cv2.imread('fake_'+test_img_path)
img1 = cv2.resize(img1, (img2.shape[0], img2.shape[1]), interpolation=cv2.INTER_AREA)

img = cv2.addWeighted(img1, 0.08, img2, 0.92, 0)
cv2.imwrite('fake_'+test_img_path, img)
```




    True




```python
import cv2
import numpy as np

img2 = cv2.imread('fake_'+test_img_path)
img1 = np.ones(shape=img2.shape, dtype='uint8')*255

img = cv2.addWeighted(img1, 0.15, img2, 0.85, 0)
cv2.imwrite('fake_'+test_img_path, img)
```




    True




```python
!python gan/infer.py --init_model output/cyclegan5/checkpoints/13/ \
                        --dataset_dir /home/aistudio/ \
                        --image_size 512 \
                        --n_samples 1 \
                        --crop_size 512 \
                        --input_style B \
                        --test_list /home/aistudio/test_last.txt \
                        --model_net CycleGAN \
                        --net_G unet_256 \
                        --g_base_dims 32 \
                        --output ./
```
