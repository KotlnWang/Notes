# 记录一些想法

## 补充知识

逆透视变化（Inverse Perspective Mapping）、Transformer

## 步骤

分点改进，对backbone进行创新；neck进行改进；transformer改进。

## 改进方向

主体思路在BEVformer基础上往轻量化，推理更快的方向进行改进。

1. 把BEVFomer中RNN的方式改为LSTM或GRU的方式；（不行）
2. 把经过backbone、fpn提取过的网络和bev特征融合；（不行）
3. 像LSS中的深度能否用diffusion的思路来生成，而不是预先设定41个1米间隔；
4. 对于bevformer加入fpn、corner pool、偏移量采用更大卷卷积核1->3、head可以采用集成的方式（anchor based and anchor free）；

#### Backbone

轻量化的模型（mobilenet）
