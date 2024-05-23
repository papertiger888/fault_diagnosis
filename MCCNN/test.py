import tensorflow as tf

# 查看 TensorFlow 版本
print("TensorFlow 版本:", tf.__version__)

# 检查是否安装了 GPU 版本
if tf.test.is_built_with_cuda():
    print("已安装 GPU 版本")
else:
    print("已安装 CPU 版本")
