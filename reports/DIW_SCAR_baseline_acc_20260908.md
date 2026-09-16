# DIW 与 SCAR 实验 baseline 和准确率汇总

整理日期：2026-09-08。以下均为本地保存的实验结果，不是论文原表数值。Acc 单位为 %，越高越好。不同批次分别列出，不能把某一批次的方法结果替换进另一批次后直接比较。

## 1. DIW：完整 baseline 主表

这里 DIW 指 IWPU 的动态交替训练版本，与 SCAR 实验中的 ADIW-SCAR 分属不同任务。主表含 8 个 baseline：tePU、trPU、ftPU、mtPU、mtsPU、DAPU、UDAPU、GIW。每格为 3 种目标 PU 数据量 × 10 个种子，共 30 次结果的均值 ± 样本标准差；波动同时包含目标数据量与随机种子的影响。IO = input-output relation shift；Support = support shift。

**版本说明：**图像任务采用 7 月主表批次，Diabetes 和 FoodStamp 采用 7 月 21 日修正预处理后的重跑结果。后续每域 batch 语义修正仅在 DIW/two-step 专项比较中重跑，见第 2 节；本节保留其他 baseline 实际完成的历史批次，不能当作全部使用最终协议的统一重跑。

| 方法 | MNIST IO | MNIST Support | FMNIST IO | FMNIST Support | CIFAR-10 IO | CIFAR-10 Support | Diabetes |
|---|---:|---:|---:|---:|---:|---:|---:|
| IWPU / DIW（研究方法） | 70.17 ± 8.96 | 75.21 ± 4.67 | 91.68 ± 4.86 | 96.65 ± 1.32 | 70.10 ± 4.01 | 82.71 ± 4.01 | 71.52 ± 1.47 |
| tePU | 72.99 ± 6.69 | 73.24 ± 5.02 | 88.54 ± 3.93 | 96.01 ± 1.83 | 67.64 ± 6.21 | 66.87 ± 8.56 | 62.43 ± 4.62 |
| trPU | 55.51 ± 1.82 | 58.82 ± 3.31 | 73.21 ± 7.51 | 90.97 ± 1.29 | 66.79 ± 5.48 | 82.92 ± 2.88 | 71.39 ± 4.21 |
| ftPU | 62.34 ± 6.80 | 67.30 ± 4.71 | 69.80 ± 6.89 | 91.79 ± 2.81 | 64.73 ± 3.53 | 83.95 ± 2.42 | 71.98 ± 1.67 |
| mtPU | 70.31 ± 8.09 | 73.66 ± 3.98 | 90.82 ± 4.12 | 95.43 ± 1.98 | 69.94 ± 4.22 | 82.22 ± 3.68 | 72.08 ± 1.30 |
| mtsPU | 69.63 ± 8.65 | 74.48 ± 4.20 | 90.76 ± 4.75 | 95.83 ± 1.64 | 69.88 ± 4.34 | 82.15 ± 3.96 | 71.81 ± 1.26 |
| DAPU | 69.52 ± 8.98 | 75.68 ± 4.48 | 91.27 ± 4.29 | 95.78 ± 1.51 | 70.68 ± 4.22 | 83.64 ± 3.40 | 71.96 ± 1.32 |
| UDAPU | 55.55 ± 1.93 | 64.90 ± 4.49 | 74.23 ± 8.14 | 93.34 ± 2.37 | 66.84 ± 5.36 | 84.17 ± 2.88 | 69.95 ± 6.88 |
| GIW（近似实现） | 71.50 ± 7.38 | 73.72 ± 6.13 | 88.50 ± 3.83 | 95.44 ± 2.00 | 70.75 ± 3.62 | 80.96 ± 4.33 | 55.64 ± 9.62 |

| Baseline | 本次实现的含义 |
|---|---|
| tePU | 仅用目标域 PU 数据训练 |
| trPU | 仅用源域 PU 数据训练 |
| ftPU | 源域预训练，再用目标域 PU 数据微调 |
| mtPU | 共享特征、源/目标两个分类头，组合两域 PU 风险 |
| mtsPU | 单个共享分类器，组合源/目标 PU 风险 |
| DAPU | 两域 PU 风险 + MMD 特征对齐 |
| UDAPU | 源域 PU 风险 + 目标域无标签 MMD 对齐 |
| GIW | 先构造伪 PN 标签，再做动态重要性加权；本项目为近似实现 |

FoodStamp 是附录任务（地理区域偏移），修正重跑结果如下：

| 方法 | FoodStamp Acc (%) |
|---|---:|
| IWPU / DIW（研究方法） | 69.54 ± 1.78 |
| tePU | 59.41 ± 4.98 |
| trPU | 70.84 ± 1.24 |
| ftPU | 69.01 ± 1.92 |
| mtPU | 69.38 ± 1.42 |
| mtsPU | 69.56 ± 1.54 |
| DAPU | 69.61 ± 1.84 |
| UDAPU | 70.51 ± 1.10 |
| GIW（近似实现） | 68.60 ± 1.98 |

来源：[DIW 图像任务主表](<C:/Users/Tamsles/Documents/Codex/2026-07-15/nin/outputs/iwpu_reproduction/reports/main_table.csv>)。

来源：[Diabetes 修正主表](<C:/Users/Tamsles/Documents/Codex/2026-07-15/nin/outputs/iwpu_reproduction/reports/tabular_no_target_rescale_v2/main_table.csv>)。

来源：[FoodStamp 修正表](<C:/Users/Tamsles/Documents/Codex/2026-07-15/nin/outputs/iwpu_reproduction/reports/tabular_no_target_rescale_v2/appendix_foodstamp.csv>)。

来源：[baseline 定义](<C:/Users/Tamsles/Documents/Codex/2026-07-15/nin/outputs/iwpu_reproduction/reports/../README.md>)。

## 2. DIW 与传统 two-step：修正版正式配对实验

420 个结果、210 个配对单元；每任务为 (10P,50U)、(20P,100U)、(40P,200U) × seeds 0–9。每格均值 ± 样本标准差。two-step 先训练完整比率模型，再冻结逐样本权重训练分类器；使用修正后的每域总 batch=256。旧 two-step 数值已被这一批次替代。

| 任务 | DIW / IWPU | two-step baseline | DIW − two-step (pp) |
|---|---:|---:|---:|
| cifar10_io | 69.78 ± 4.44 | 67.60 ± 5.38 | +2.18 |
| cifar10_support | 83.17 ± 4.21 | 73.17 ± 7.02 | +10.00 |
| diabetes | 71.72 ± 1.20 | 64.54 ± 5.98 | +7.18 |
| fmnist_io | 91.22 ± 5.59 | 89.75 ± 3.76 | +1.47 |
| fmnist_support | 96.35 ± 1.65 | 95.82 ± 3.00 | +0.53 |
| mnist_io | 69.72 ± 7.77 | 72.05 ± 6.76 | -2.33 |
| mnist_support | 75.73 ± 4.70 | 74.68 ± 5.55 | +1.05 |
| 全部任务等权汇总（均值） | 79.67 | 76.80 | +2.87 |

本批次总体 DIW 高 2.87 个百分点；优势不覆盖全部任务（MNIST IO 中 two-step 更高）。属于独立 from-paper 实现，不能称为作者官方复现。
来源：[最终逐配对准确率](<C:/Users/Tamsles/Documents/Codex/2026-07-15/nin/outputs/iwpu_reproduction/reports/diw_vs_two_step_paper_v1_final/paired_cells.csv>)。

来源：[协议修正记录](<C:/Users/Tamsles/Documents/Codex/2026-07-15/nin/outputs/iwpu_reproduction/reports/diw_two_step_reaudit.md>)。

## 3. SCAR：原始域数据 baseline 对照

MNIST、MLP、目标域固定旋转 30°；源域弱标注数据 59,000，目标域弱标注数据 1,000，独立目标域测试集 10,000。200 epochs，seeds 0–4。每个种子取最后 10 个 epoch 的目标分类准确率均值，再报告 5 个种子的均值 ± 总体标准差。

| 方法 | 含义 | Acc (%) |
|---|---|---:|
| trainSCAR | 仅源域弱标注数据 | 54.30 ± 0.86 |
| testSCAR | 仅目标域弱标注数据 | 38.82 ± 2.44 |
| ftSCAR / pooled SCAR | 直接混合源域和目标域，不使用 ADIW | 60.31 ± 1.68 |
| ADIW-SCAR | 两域数据 + ADIW 权重 | 60.98 ± 2.44 |
| Conditioned-Ratio-SCAR | 条件化神经比率估计器 | 60.65 ± 2.84 |
| Multihead-Ratio-SCAR | 多头神经比率估计器 | 60.99 ± 1.85 |

注意：本仓库的 ftSCAR 是 pooled SCAR 的别名，不能按 ftPU 的含义解释成“先预训练后微调”。ADIW 相对 pooled 的均值增益仅 0.67 pp，原报告配对 p=0.417，尚不能称为明确提升。
来源：[SCAR 原始 baseline 及逐种子结果](<C:/Users/Tamsles/Documents/SCAR/outputs/scarce_comparison/original_ABLATION_REPORT.md>)。

来源：[含神经比率方法的准确率表](<C:/Users/Tamsles/Documents/SCAR/outputs/scarce_comparison/scar_method_comparison.csv>)。

## 4. SCAR：后续独立批次（补充，勿与第 3 节混表）

Phase A 在统一 runner 中比较 no-IW、pooled、ADIW 与神经比率方法。下面每格为 5 种子均值 ± 总体标准差，仍使用最后 10 个 epoch 的目标准确率。

| 方法 | 旋转 10° | 旋转 20° | 旋转 30° |
|---|---:|---:|---:|
| no_iw | 86.16 ± 0.76 | 72.37 ± 1.25 | 54.59 ± 2.35 |
| pooled_scar | 86.66 ± 0.52 | 75.37 ± 1.57 | 59.55 ± 2.98 |
| adiw | 87.06 ± 0.28 | 76.15 ± 1.39 | 62.69 ± 2.88 |
| unified | 85.91 ± 0.80 | 75.58 ± 1.14 | 60.66 ± 1.27 |
| multihead | 85.96 ± 0.76 | 74.96 ± 1.04 | 59.66 ± 3.52 |
| fusion | 86.38 ± 0.55 | 75.94 ± 0.58 | 59.36 ± 1.90 |
| separate_full | 未列入该批次 | 未列入该批次 | 52.01 ± 2.63 |

来源：[Phase A 分设置统计](<C:/Users/Tamsles/Documents/SCAR/outputs/wisteria_next_round_phase_a/summary/summary_by_setting.csv>)。

Phase B0 的训练流程复核：MNIST 旋转 30°，10 种子，均值 ± 总体标准差。

| 方法 / 干预 | Acc (%) |
|---|---:|
| adiw_mean | 60.99 ± 2.33 |
| adiw_ones | 61.09 ± 2.45 |
| adiw_original | 61.79 ± 1.85 |
| adiw_shuffle | 60.47 ± 1.90 |
| current_no_iw | 53.70 ± 1.79 |
| matched_no_iw | 60.29 ± 2.01 |

current_no_iw 是当前未匹配流程；matched_no_iw 匹配 ADIW 的训练流程但不使用 IW；adiw_ones 将权重置 1；adiw_mean 用均值权重；adiw_shuffle 打乱样本与权重对应关系；adiw_original 使用原始权重。该批次表明训练流程差异会明显影响 baseline，不能仅依据跨批次 acc 判断重要性加权的收益。
来源：[Phase B0 报告](<C:/Users/Tamsles/Documents/SCAR/reports/phase_b0_adiw_causal_decomposition.md>)。

## 5. 核对范围与使用说明

- 已检查 DIW 主表 63 个方法×任务条目，每项 30 次；Diabetes/FoodStamp 使用预处理修正版。
- 已从最终配对 CSV 独立重算 DIW/two-step 均值和标准差，核对 210 个唯一配对键、每任务 30 个。
- 已从 SCAR 逐种子数据独立重算第 3 节均值与总体标准差；B0 每方法 10 次，数值由正式 runs/B0 下逐次 summary.json 重算。
- DIW 的 std 为样本标准差；SCAR 历史报告使用总体标准差，两者均已明确标注。没有将域判别器 accuracy 或 ratio RMSE 当作分类 acc。
- 本汇总聚焦 baseline 准确率；神经架构额外批次、四类偏移压力测试、Gaussian oracle、B2 clipping 网格和 B4 异质偏移保留在原始报告中，不拼成一个排行榜。
- 置信评估：可附版本与口径说明转发。历史 DIW 主表并非最终 batch 协议下全方法重跑，不支持跨版本的严格优劣结论。

来源：[本汇总的可重跑提取与核对脚本](<C:/Users/Tamsles/Documents/SCAR/reports/build_baseline_summary.ps1>)。

