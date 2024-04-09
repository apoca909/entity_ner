# punc_pytorch

## 训练
```
python cmd_run_felix.py
```

## 测试
```
python test.py
```

# 工具
  ###### 转script_model
```
transformers版本: '4.28.0.dev0'
运行: python tools/torch_jit/export_jit.py --config $model_dir/config.json --checkpoint $model_dir/pytorch_model.bin --output_file ug.final.zip
```

# 模型和测试集仓库
```
https://github.com/Speech-Nlp-Lab/punctuation_model
```
