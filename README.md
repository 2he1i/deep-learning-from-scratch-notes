# 📚 Fish-Book-Notes: 深度学习从零实现

本项目是我在学习《深度学习入门：基于Python的理论与实现》（斋藤康毅 著）过程中的**随书笔记**与**自主代码实现**。

项目的核心目标是：**不依赖外部深度学习框架（如 PyTorch、TensorFlow）**，仅使用 Python 和 NumPy 从零构建神经网络，从而深刻理解深度学习的底层原理（如反向传播、卷积运算等）。

---

## 🛠️ 环境要求与配置

本项目开发环境基于 **Windows Subsystem for Linux (WSL2)**，并使用高性能 Python 包管理器 [**uv**](https://github.com/astral-sh/uv) 进行依赖管理。

### 1. 环境
*   **OS**: WSL2 (Ubuntu 22.04+ 推荐)
*   **Package Manager**: `uv` (0.9.26+)
*   **Python**: 3.14+

### 2. 快速开始
如果你已经安装了 `uv`，可以通过以下命令快速克隆并初始化环境：

```bash
# 克隆仓库
git clone https://github.com/你的用户名/你的项目名.git
cd 你的项目名

# 创建虚拟环境并安装依赖
uv venv
source .venv/bin/activate
uv sync  # 如果你配置了 pyproject.toml 或 uv.lock
# 或者手动安装基础依赖
uv pip install numpy ...
```

---

## 🤝 声明

1.  **学习用途**：本项目仅用于个人学习和技术交流。
2.  **版权说明**：代码逻辑参考了斋藤康毅所著的《深度学习入门：基于Python的理论与实现》，部分数据集加载脚本（如 `mnist.py`）源自原书配套源码。
3.  **笔记原创**：项目中的笔记与中文注释均为个人理解整理，如有谬误欢迎指正。

---