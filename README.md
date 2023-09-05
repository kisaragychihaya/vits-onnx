# vits-onnx


## 这是一个纯洁的使用ONNXRuntime推理的VITS

### 那么，依赖是什么呢
通过pip安装如下的包即可 ```jieba pypinyin scipy ```

通过pip 安装```onnxruntime```或者```onnxruntime-gpu```或者```onnxruntime-dml```或者```onnxruntime-openvino```（选择其一即可)

### 神奇模型在哪里
1.您可以前往 [这里](https://github.com/zixiiu/Digital_Life_Server) 下载权重文件,并根据其教程配置好环境

```python setup.py build_ext --inplace```这里请替换为```python setup.py install```

然后cd 回本仓库 并使用如下命令行转换模型

```bash
pushd vits
python export_onnx.py --cfg path_to.json --checkpoint path_to.pth --onnx_model path_to.onnx
popd
```

2.您也可以稍后直接下载权重

稍后继续写
TODO
