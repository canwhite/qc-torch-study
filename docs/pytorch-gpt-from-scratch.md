# PyTorch 实战：用 GPT 模型理解 PyTorch 核心概念

> 项目来源：[novel_gpt](https://github.com/yourusername/novel_gpt) — 一个从头实现 GPT 的小说生成模型

---

## 一、项目概览

这是一个基于 PyTorch 从零实现 GPT 的项目，核心文件：

| 文件 | 职责 |
|------|------|
| `model.py` | GPT 模型定义（Transformer 架构） |
| `train.py` | 训练循环（Trainer 类） |
| `data.py` | 数据加载（Dataset、DataLoader） |
| `config.py` | 超参数配置（dataclass） |

---

## 二、PyTorch 核心概念详解

### 1. `nn.Module` — 自定义模型基类

**所有神经网络模型的基类**，需要重写 `forward()` 方法。

```python
# 来自 model.py
class GPT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()  # 必须调用父类构造
        self.config = config

        # 子模块会自动被追踪
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        # 前向传播逻辑
        logits = ...
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
```

**关键点：**
- `super().__init__()` 必须在子类构造函数中调用
- `nn.Module` 的子类属性（如 `nn.Linear`、`nn.ModuleList`）会被自动注册
- 调用 `model.parameters()` 会递归收集所有可学习参数

---

### 2. `nn.Parameter` — 可学习参数

**张量的一种特殊类型，会被 `parameters()` 自动收集。**

```python
# 来自 model.py
class LayerNorm(nn.Module):
    def __init__(self, ndim: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))  # γ 增益参数
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None  # β 偏置参数

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)
```

**等效写法：**

```python
# 手动定义（需要自己注册）
self.weight = torch.ones(ndim)           # 普通 tensor
self.register_parameter('weight', ...)     # 或用 register_parameter

# 推荐写法（nn.Parameter 自动注册）
self.weight = nn.Parameter(torch.ones(ndim))
```

---

### 3. `register_buffer` — 非学习参数

**模型附属的张量（如BatchNorm均值、因果掩码），不参与梯度计算但会随模型移动。**

```python
# 来自 model.py - 因果自注意力掩码
self.register_buffer(
    "bias",
    torch.tril(torch.ones(config.block_size, config.block_size))
        .view(1, 1, config.block_size, config.block_size)
)
```

**特点：**
- 不参与梯度计算（不更新）
- 随 `.to(device)` 一起迁移设备
- 可通过 `model.bias` 访问

---

### 4. `nn.ModuleDict` / `nn.ModuleList` — 容器管理

```python
# ModuleDict - 按键索引
self.transformer = nn.ModuleDict(dict(
    wte = nn.Embedding(...),   # token embedding
    wpe = nn.Embedding(...),   # position embedding
    h = nn.ModuleList([...]),   # transformer blocks
    ln_f = LayerNorm(...),
))

# 访问方式
tok_emb = self.transformer.wte(idx)      # 按键访问
x = self.transformer.h[0](x)            # 按索引访问
```

---

### 5. `torch.no_grad()` — 推断模式

**关闭梯度计算，加快推理速度、减少内存占用。**

```python
# 来自 model.py
@torch.no_grad()
def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    """文本生成"""
    for _ in range(max_new_tokens):
        logits, _ = self(idx_cond)           # 无梯度追踪
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

# 来自 train.py
@torch.no_grad()
def evaluate(self, max_batches=None):
    self.model.eval()
    for batch in self.val_loader:
        _, loss = self.model(x, y)
        total_loss += loss.item()
    self.model.train()
    return total_loss / num_batches
```

**配合使用：**
- `.eval()` — 切换模型为评估模式（影响 Dropout、BatchNorm 行为）
- `.train()` — 切换回训练模式

---

### 6. `nn.functional` — 函数式 API

**无需维护状态的函数，如激活函数、损失函数。**

```python
import torch.nn.functional as F

# 损失函数
loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

# 激活函数
att = F.softmax(att, dim=-1)
x = self.gelu(x)

# LayerNorm
x = F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)
```

---

### 7. `nn.Dropout` — 正则化

```python
# 训练时随机置零（防止过拟合）
self.attn_dropout = nn.Dropout(config.dropout)
self.resid_dropout = nn.Dropout(config.dropout)

# forward 中使用
att = self.attn_dropout(att)    # 训练时有 dropout 效果
y = self.resid_dropout(self.c_proj(y))
```

- `.train()` 模式：激活 Dropout
- `.eval()` 模式：Dropout 关闭（所有神经元都参与）

---

### 8. `nn.Embedding` — 查表层

**将 token ID 映射为密集向量。**

```python
# 来自 model.py
self.transformer.wte = nn.Embedding(config.vocab_size, config.n_embd)   # token embedding
self.transformer.wpe = nn.Embedding(config.block_size, config.n_embd)  # position embedding

# forward 中使用
tok_emb = self.transformer.wte(idx)           # (B, T) -> (B, T, n_embd)
pos_emb = self.transformer.wpe(pos)           # (T,) -> (T, n_embd)
x = tok_emb + pos_emb
```

---

### 9. 权重共享

```python
# 来自 model.py
self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
# 输入输出嵌入共享权重（节省参数）
self.transformer.wte.weight = self.lm_head.weight
```

---

## 三、训练流程（train.py 解析）

### 1. 模型创建与设备迁移

```python
from .model import create_model

# create_model 内部实现
def create_model(config, device):
    model = GPT(config)
    model = model.to(device)    # 将模型参数迁移到 GPU/MPS
    return model
```

### 2. `optim` — 优化器

```python
# AdamW 优化器（带权重衰减的 Adam）
self.optimizer = torch.optim.AdamW(
    self.model.parameters(),      # 优化所有可学习参数
    lr=config.training.learning_rate,
    betas=(0.9, 0.95),           # 动量参数
    weight_decay=0.01,            # 权重衰减
)
```

### 3. `lr_scheduler` — 学习率调度

```python
# LambdaLR - 自定义调度函数
def lr_lambda(step):
    if step < warmup_steps:
        return step / warmup_steps        # 线性预热
    return max(0.0, 1.0 - step / max_steps)  # 线性衰减

self.scheduler = torch.optim.lr_scheduler.LambdaLR(
    self.optimizer,
    lr_lambda
)

# 训练循环中
for step in range(max_steps):
    optimizer.step()    # 更新参数
    scheduler.step()    # 更新学习率
```

### 4. 训练步骤

```python
def train_step(self, batch):
    x, y = batch
    x, y = x.to(self.device), y.to(self.device)

    self.optimizer.zero_grad()     # 清除上一步梯度
    _, loss = self.model(x, targets=y)  # 前向 + 计算损失
    loss.backward()                # 反向传播（计算梯度）

    # 梯度裁剪 - 防止梯度爆炸
    torch.nn.utils.clip_grad_norm_(
        self.model.parameters(),
        self.config.training.grad_clip,  # 通常为 1.0
    )

    self.optimizer.step()          # 更新参数
    self.scheduler.step()          # 更新学习率
    return loss.item()
```

### 5. 完整训练循环

```python
def train(self):
    self.model.train()              # 切换训练模式

    for step in pbar:
        batch = next(data_iter)
        loss = self.train_step(batch)

        # 定期评估
        if step % eval_interval == 0:
            val_loss = self.evaluate()
            self.model.train()

        # 保存检查点
        if step % save_interval == 0:
            self.save_checkpoint(f"step_{step}.pt")
```

---

## 四、数据加载（data.py 解析）

### 1. `Dataset` — 数据集抽象

```python
# 来自 data.py
class TokenDataset(Dataset):
    def __init__(self, tokens: torch.Tensor, block_size: int):
        self.tokens = tokens
        self.block_size = block_size

    def __len__(self):
        # 数据集样本总数
        return max(0, len(self.tokens) - self.block_size)

    def __getitem__(self, idx):
        # 返回单个样本 (输入, 目标)
        x = self.tokens[idx:idx + self.block_size]
        y = self.tokens[idx + 1:idx + self.block_size + 1]
        return x, y
```

### 2. `DataLoader` — 批量加载

```python
# 来自 data.py
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,           # 每个 epoch 打乱数据
    num_workers=0,          # Mac M2 上 0 更稳定
    pin_memory=False,       # MPS 不支持 pin_memory
)

# 使用方式
for x, y in train_loader:
    # x, y 都是 batch_size 大小的张量
    loss = model(x, y)
```

### 3. 数据划分

```python
# 90% 训练 / 10% 验证
split_idx = int(len(tokens) * 0.9)
train_tokens = tokens[:split_idx]
val_tokens = tokens[split_idx:]
```

---

## 五、模型持久化

### 1. `torch.save` — 保存检查点

```python
# 来自 model.py
def save_checkpoint(self, path, optimizer=None, step=0):
    checkpoint = {
        "model": self.state_dict(),      # 模型参数
        "config": self.config.__dict__,  # 配置文件
        "step": step,
    }
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()  # 优化器状态
    torch.save(checkpoint, path)
```

### 2. `torch.load` — 加载检查点

```python
# 来自 model.py
@classmethod
def load_checkpoint(cls, path, device="cpu"):
    checkpoint = torch.load(path, map_location=device)  # 映射到目标设备
    config = ModelConfig(**checkpoint["config"])
    model = cls(config)
    model.load_state_dict(checkpoint["model"])  # 加载参数
    model = model.to(device)
    return model, checkpoint.get("step", 0)
```

---

## 六、设备管理（MPS/CUDA/CPU）

```python
# 来自 config.py
device = "auto"
if device == "auto":
    if torch.backends.mps.is_available():   # Apple M2 GPU
        device = "mps"
    elif torch.cuda.is_available():          # NVIDIA GPU
        device = "cuda"
    else:
        device = "cpu"

# 张量迁移设备
x = x.to(device)
model = model.to(device)

# 检查设备
print(x.device)   # mps:0 / cuda:0 / cpu
```

---

## 七、概念速查表

| 概念 | 代码位置 | 说明 |
|------|---------|------|
| 自定义模型 | `model.py: GPT` | 继承 `nn.Module`，重写 `forward` |
| 可学习参数 | `model.py: LayerNorm` | `nn.Parameter(torch.ones(ndim))` |
| 非学习参数 | `model.py: register_buffer` | 因果掩码等 |
| 容器管理 | `model.py: ModuleDict/ModuleList` | 组织子模块 |
| 推断模式 | `model.py: @torch.no_grad()` | 推理时不计算梯度 |
| Dropout | `model.py: nn.Dropout` | 训练时随机丢弃，正则化 |
| Embedding | `model.py: nn.Embedding` | Token ID → 向量 |
| 优化器 | `train.py: AdamW` | 更新模型参数 |
| 学习率调度 | `train.py: LambdaLR` | 动态调整学习率 |
| 梯度裁剪 | `train.py: clip_grad_norm_` | 防止梯度爆炸 |
| Dataset | `data.py: TokenDataset` | 数据集抽象 |
| DataLoader | `data.py: DataLoader` | 批量加载数据 |
| 保存/加载 | `model.py: save/load_checkpoint` | 模型持久化 |
| 设备迁移 | `model.py: .to(device)` | CPU/GPU/MPS 切换 |
