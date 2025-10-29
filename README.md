# 环境配置

激活python，执行以下命令以获得相同环境：

```shell
pip install -r requirements.txt
```

# 运行说明

可以直接通过以下代码进行训练（随机种子已经在代码中指定为42）：

```bash
bash scripts/run.sh
```

或者可以打开终端到src目录下，激活相应python环境，运行以下命令：

```shell
python Train.py
```

单卡-NVIDIA GeForce RTX 3090训练约100分钟
