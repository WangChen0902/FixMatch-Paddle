# Linux端基础训练推理功能测试

Linux端基础训练推理功能测试的主程序为`test_train_inference_python.sh`，可以测试基于Python的模型训练、评估、推理等基本功能，包括裁剪、量化、蒸馏。

## 1. 测试结论汇总

- 训练相关：

| 算法名称 | 模型名称 | 单机单卡 | 单机多卡 | 多机多卡 | 模型压缩（单机多卡） |
|  :----: |   :----:  |    :----:  |  :----:   |  :----:   |  :----:   |
| FixMatch | FixMatch | 正常训练 | - | - | - |


- 推理相关：基于训练是否使用量化，可以将训练产出的模型可以分为`正常模型`和`量化模型`，这两类模型对应的推理功能汇总如下，

| 算法名称 | 模型名称 | 模型类型 |device | batchsize | tensorrt | mkldnn | cpu多线程 |
|  :----:   |  :----: |   ----   |  :----:  |   :----:   |  :----:  |   :----:   |  :----:  |
| FixMatch | FixMatch |  正常模型 | GPU | 1/1 | - | - | - |
| FixMatch | FixMatch | 正常模型 | CPU | 1/1 | - | fp32 | 支持 |


## 2. 测试流程

### 2.1 准备数据

本项目数据为目录中的`./data/sort-of-clevr.pickle`。

### 2.2 准备环境


- 安装PaddlePaddle >= 2.1
- 安装AlexNet依赖
    ```
    pip3 install  -r ../requirements.txt
    ```
- 安装AutoLog（规范化日志输出工具）
    ```
    git clone https://github.com/LDOUBLEV/AutoLog
    cd AutoLog
    pip install -r requirements.txt
    python setup.py bdist_wheel
    pip3 install ./dist/auto_log-1.0.0-py3-none-any.whl
    cd ../
    ```

### 2.3 功能测试


本模型测试方法如下所示。

```bash
bash test_tipc/test_train_inference_python.sh test_tipc/configs/FixMatch/train_infer_python.txt lite_train_lite_infer
```

输出结果如下。

```
[33m Run successfully with command - python3.7 main.py --device=gpu  --output-dir=./test_tipc/output/norm_train_gpus_1_autocast_null --epochs=1     --batch-size=64     !  [0m
[33m Run successfully with command - python3.7 tools/export_model.py --pretrained=./test_tipc/output/norm_train_gpus_1_autocast_null/epoch_RN_01.pdparams --save-inference-dir=./test_tipc/output/norm_train_gpus_1_autocast_null!  [0m
[33m Run successfully with command - python3.7 tools/infer.py --use-gpu=True     --model-dir=./test_tipc/output/norm_train_gpus_1_autocast_null/ --batch-size=1   --benchmark=False     > ./test_tipc/output/python_infer_gpu_usetrt_null_precision_null_batchsize_1.log 2>&1 !  [0m
[33m Run successfully with command - python3.7 tools/infer.py --use-gpu=False     --model-dir=./test_tipc/output/norm_train_gpus_1_autocast_null/ --batch-size=1     --benchmark=False     > ./test_tipc/output/python_infer_cpu_usemkldnn_null_threads_null_precision_null_batchsize_1.log 2>&1 !  [0m
  
```


## 3. 更多教程

本文档为功能测试用，更丰富的训练预测使用教程请参考：  

* [模型训练、预测、推理教程](../../README.md)  
