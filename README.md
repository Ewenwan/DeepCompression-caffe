# DeepCompression-caffe

Caffe for Deep Compression

使用Caffe实现，需要加入一个mask来表示剪枝。剪枝的阈值，是该layer的权重标准差乘上某个超参数。

[参考](https://xmfbit.github.io/2018/10/03/paper-summary-model-pruning/#more)
# 简介
      1.在.cu中目前仍然是调用cpu_data接口，所以可能会增加与gpu数据交换的额外耗时，这个不影响使用，
        后面慢慢优化。~(已解决) 
      2.目前每层权值修剪的比例仍然是预设的，这个比例需要迭代试验以实现在尽可能压缩权值的同时保证精度。
        所以如何自动化选取阈值就成为了后面一个继续深入的课题。 
      3.直接用caffe跑出来的模型依然是原始大小，因为模型依然是.caffemodel类型，
        虽然大部分权值为0且共享，但每个权值依然以32float型存储，
         故后续只需将非零权值及其下标以及聚类中心存储下来即可，这部分可参考作者论文，写的很详细。 
      4.权值压缩仅仅压缩了模型大小，但在前向inference时不会带来速度提升。
        因此，想要在移动端做好cnn部署，就需要结合小的模型架构、模型量化以及NEON指令加速等方法来实现。 

# Introduction
This is a simple caffe implementation of Deep Compression(https://arxiv.org/abs/1510.00149), including weight prunning and quantization.<br>
According to the paper, the compression are implemented only on convolution and fully-connected layers.<br>
Thus we add a CmpConvolution and a CmpInnerProduct layer.<br>
The params that controlls the sparsity including:<br>
* sparse_ratio: the ratio of pruned weights<br>
* class_num: the numbers of k-means for weight quantization<br>
* quantization_term: whether to set quantization on <br>

For a better understanding, please see the examples/mnist and run the demo script, which automatically compresses a pretrained MNIST LeNet caffemodel.

# Run LeNet Compression Demo

```
$Bash
```

```Bash
# clone repository and make 
$ git clone https://github.com/may0324/DeepCompression-caffe.git
$ cd DeepCompression-caffe
$ make -j 32 

# run demo script, this will finetune a pretrained model
$ python examples/mnist/train_compress_lenet.py

```

# Details 
the sparse parameters of lenet are set based on the paper as follows:<br>

|    layer name   |      sparse ratio     |           quantization num              |
| :------------- |:-------------:| :-----:|
| conv1           |               0.33               |                256               |
| conv2           |               0.8                |                256               |
| fc1             |               0.9                |                32                |
| fc2             |               0.8                |                32                |    

In practice, the layers are much more sensitive to weight prunning than weight quantization. <br>
So we suggest to do weight prunning layer-wisely 
and do weight quantization finally since it almost does no harm to accuary. <br>
In the script demo, we set the sparse ratio (the ratio of pruned weights) layer-wisely and do each finetuning iteration.
After all layers are properly pruned, weight quantization are done on all layers simultaneously. <br>

The final accuracy of finetuned model is about 99.06%, you can check if the weights are most pruned and weight-shared for sure.<br>

# Model Size
The size of finetuned model is still the same as the original one since it is stored in 'caffemodel' format. Although most of the weights are pruned and shared, the weights are still stored in float32. You can only store the non-zero weight and cluster center to reduce the redundacy of finetuned model, please refer to the paper.

Please refer to http://blog.csdn.net/may0324/article/details/52935869 for more. <br>
Enjoy! 

