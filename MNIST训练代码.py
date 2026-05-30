"""
项目1：手写数字识别 - MNIST + LeNet-5
"""

import paddle
import numpy as np
from paddle.vision import datasets, models, transforms
from paddle.vision.transforms import Normalize
from collections import defaultdict

print('PaddlePaddle version:', paddle.__version__)

# ── 数据预处理 ──────────────────────────────────────────────────────────
print('\n加载 MNIST 数据集...')
transform = Normalize(mean=[127.5], std=[127.5], data_format='CHW')

train_dataset = datasets.MNIST(mode='train', transform=transform, download=True)
test_dataset  = datasets.MNIST(mode='test',  transform=transform, download=True)

print(f'训练集: {len(train_dataset)} 张 | 测试集: {len(test_dataset)} 张')

# ── 模型组网 ────────────────────────────────────────────────────────────
print('\n构建 LeNet-5 模型...')
lenet = models.LeNet(num_classes=10)
model = paddle.Model(lenet)

# ── 模型训练配置 ──────────────────────────────────────────────────────
model.prepare(
    optimizer=paddle.optimizer.Adam(parameters=model.parameters()),
    loss=paddle.nn.CrossEntropyLoss(),
    metrics=paddle.metric.Accuracy()
)

# ── 模型训练 ────────────────────────────────────────────────────────────
print('\n' + '='*60)
print('开始训练（5 epochs, batch_size=64）...')
print('='*60 + '\n')

model.fit(
    train_dataset,
    epochs=5,
    batch_size=64,
    verbose=1
)

# ── 模型评估 ────────────────────────────────────────────────────
print('\n' + '='*60)
print('内置评估...')
print('='*60)

builtin_results = model.evaluate(
    test_dataset,
    batch_size=64,
    verbose=1
)

test_acc = builtin_results['acc']
error_rate = 1.0 - test_acc

print(f'\n内置测试集准确率 (Accuracy): {test_acc:.4f} ({test_acc*100:.2f}%)')
print(f'内置测试集错误率 (Error-rate): {error_rate:.4f} ({error_rate*100:.2f}%)')

# ── 计算 Recall 和 Precision ────────────────────────────────────────────
print('\n计算 Macro Recall 和 Macro Precision...')

all_preds = []
all_targets = []

# 用 model.predict_batch 预测整个测试集
print('预测测试集...')
for i, (img, label) in enumerate(test_dataset):
    img_batch = np.expand_dims(img.astype('float32'), axis=0)
    out = model.predict_batch(img_batch)[0]
    pred = out.argmax()
    all_preds.append(pred)
    all_targets.append(int(label))  # 转为 int，避免 numpy 类型问题
    
    if (i + 1) % 2000 == 0:
        print(f'  已预测 {i+1}/{len(test_dataset)} 张图片...')

print(f'预测完成，共 {len(all_preds)} 个样本')

# 计算 TP/FP/FN
tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
for p, t in zip(all_preds, all_targets):
    if p == t:
        tp[t] += 1
    else:
        fp[p] += 1
        fn[t] += 1

recalls, precisions = [], []
for c in range(10):
    r = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
    p = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
    recalls.append(r)
    precisions.append(p)

macro_recall    = sum(recalls) / 10
macro_precision = sum(precisions) / 10

print(f'\n{"="*60}')
print(f'  Macro Recall:    {macro_recall:.4f}')
print(f'  Macro Precision: {macro_precision:.4f}')
print(f'  Accuracy:       {test_acc:.4f}')
print(f'  Error-rate:     {error_rate:.4f}')
print(f'{"="*60}')

# ── 输出结果 ────────────────────────────────────────
print('\n┌─────────────────────────────────────────────────────────┐')
print('│  项目1实验结果                            │')
print('├─────────────────────────────────────────────────────────┤')
print(f'│  Accuracy:       {test_acc:.4f}  ({test_acc*100:.2f}%)                   │')
print(f'│  Error-rate:     {error_rate:.4f}  ({error_rate*100:.2f}%)                 │')
print(f'│  Macro Recall:   {macro_recall:.4f}                                      │')
print(f'│  Macro Precision:{macro_precision:.4f}                                      │')
print('└─────────────────────────────────────────────────────────┘')

# ── 保存结果到 JSON ─────────────────────────────────
import json
import os

result = {
    'accuracy': float(test_acc),
    'error_rate': float(error_rate),
    'macro_recall': float(macro_recall),
    'macro_precision': float(macro_precision)
}

result_path = os.path.join('.', 'mnist_result.json')
with open(result_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print(f'\n结果已保存至: {os.path.abspath(result_path)}')

