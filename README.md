# 环境配置

激活python，执行以下命令以获得相同环境：

```shell
pip install -r requirements.txt
```

# 数据集

采用的是IWSLT2017的英语德语数据集，数据已经放在仓库中，可下载查看

# 运行说明

可以直接通过以下代码进行训练（随机种子指定为42）：

```bash
bash scripts/run.sh
```

或者可以打开终端到src目录下，激活相应python环境，运行以下命令：

```shell
python Train.py
```

单卡-NVIDIA GeForce RTX 3090训练时长约100分钟，最终得到的模型的权重参数保存在best_transformer_model.pth中，并生成训练和验证损失曲线保存在results文件夹下，该文件可通过链接查看并下载：

```
通过网盘分享的文件：best_transformer_model.pth
链接: https://pan.baidu.com/s/1MQ_qzvO0-fGUQRTMkbdF2w 提取码: 71cq 
--来自百度网盘超级会员v5的分享
```

