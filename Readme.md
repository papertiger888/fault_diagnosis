### 原始数据及说明
1. data文件夹存储了原始数据
2. 其中0HP、1HP、2HP、3HP分别为不同负载下采集的数据、padelibao文件夹代表PU数据集
3. 12_drive代表驱动端数据、B、IR、OR、normal分别代表滚珠故障、内圈故障、外圈故障、正常轴承

### MCCNN代码说明及运行环境
1. MCCNN文件夹包含MCCNN模型需要的运行程序及其他工具文件；可通过运行1DCNN_SVD主程序运行
2. MCCNN的运行环境的深度学习框架为tensorflow，安装如下
官网：https://tensorflow.google.cn/?hl=zh-cn
创建深度学习环境并命名为tf2，指定python版本为3.11:
conda create --name tf2 python=3.11
Anaconda激活tf2环境:
conda activate tf2
Anaconda退出激活环境:
conda deactivate
安装Tensorflow并指定版本为2.15，使用豆瓣源进行加速
pip install tensorflow==2.15.0 -i https://pypi.douban.com/simple/
安装scikit-learn，使用豆瓣源进行加速
pip install scikit-learn -i https://pypi.douban.com/simple/

### MoE-DANN代码说明及运行环境
1. 通过train_moe.sh运行
2. 运行环境如下
* Pytorch 0.3/0.4
* sklearn
* termcolor

### 实验结果文件夹result

1. ./result/MCCNN-result 存储MCCNN实验结果
2. ./result/MoE-DANN-result 存储MoE-DANN实验结果
3. ./result/different_load 存储同种故障下不同负载波形图
4. ./result/different_noise 存储了不同噪声下的波形图

### 对比模型可参考
https://github.com/timeseriesAI/tsai
https://github.com/AaronCosmos/wdcnn_bearning_fault_diagnosis
https://github.com/hfawaz/dl-4-tsc