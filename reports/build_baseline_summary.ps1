$ErrorActionPreference = 'Stop'
$diw = 'C:/Users/Tamsles/Documents/Codex/2026-07-15/nin/outputs/iwpu_reproduction/reports'
$scar = 'C:/Users/Tamsles/Documents/SCAR'
$out = "$scar/reports/DIW_SCAR_baseline_acc_20260908.md"
$lines = [System.Collections.Generic.List[string]]::new()
function Add([string]$s) { $lines.Add($s) }
function Stats($values, [int]$ddof = 0) {
    $v = @($values | ForEach-Object { [double]$_ })
    if ($v.Count -le $ddof) { throw 'Insufficient observations' }
    $m = ($v | Measure-Object -Average).Average
    $ss = ($v | ForEach-Object { ($_ - $m) * ($_ - $m) } | Measure-Object -Sum).Sum
    return @($m, [Math]::Sqrt($ss / ($v.Count - $ddof)))
}
function Fmt($m, $s, $scale = 1) { return ('{0:F2} ± {1:F2}' -f ([double]$m * $scale), ([double]$s * $scale)) }
function Source([string]$label, [string]$path) { Add "来源：[$label](<$path>)。"; Add '' }
$main = @(Import-Csv "$diw/main_table.csv" | Where-Object dataset -ne 'diabetes') + @(Import-Csv "$diw/tabular_no_target_rescale_v2/main_table.csv")
$food = @(Import-Csv "$diw/tabular_no_target_rescale_v2/appendix_foodstamp.csv")
$methods = [ordered]@{ours='IWPU / DIW（研究方法）';tepu='tePU';trpu='trPU';ftpu='ftPU';mtpu='mtPU';mtspu='mtsPU';dapu='DAPU';udapu='UDAPU';giw='GIW（近似实现）'}
$tasks = @('mnist/io','mnist/support','fmnist/io','fmnist/support','cifar10/io','cifar10/support','diabetes/race')
if ($main.Count -ne 63 -or @($main | Where-Object count -ne 30).Count -ne 0) { throw 'DIW main table coverage mismatch' }
Add '# DIW 与 SCAR 实验 baseline 和准确率汇总'
Add ''
Add '整理日期：2026-09-08。以下均为本地保存的实验结果，不是论文原表数值。Acc 单位为 %，越高越好。不同批次分别列出，不能把某一批次的方法结果替换进另一批次后直接比较。'
Add ''
Add '## 1. DIW：完整 baseline 主表'
Add ''
Add '这里 DIW 指 IWPU 的动态交替训练版本，与 SCAR 实验中的 ADIW-SCAR 分属不同任务。主表含 8 个 baseline：tePU、trPU、ftPU、mtPU、mtsPU、DAPU、UDAPU、GIW。每格为 3 种目标 PU 数据量 × 10 个种子，共 30 次结果的均值 ± 样本标准差；波动同时包含目标数据量与随机种子的影响。IO = input-output relation shift；Support = support shift。'
Add ''
Add '**版本说明：**图像任务采用 7 月主表批次，Diabetes 和 FoodStamp 采用 7 月 21 日修正预处理后的重跑结果。后续每域 batch 语义修正仅在 DIW/two-step 专项比较中重跑，见第 2 节；本节保留其他 baseline 实际完成的历史批次，不能当作全部使用最终协议的统一重跑。'
Add ''
Add '| 方法 | MNIST IO | MNIST Support | FMNIST IO | FMNIST Support | CIFAR-10 IO | CIFAR-10 Support | Diabetes |'
Add '|---|---:|---:|---:|---:|---:|---:|---:|'
foreach ($key in $methods.Keys) {
    $cells = @($methods[$key])
    foreach ($task in $tasks) {
        $p = $task.Split('/')
        $r = @($main | Where-Object { $_.dataset -eq $p[0] -and $_.shift -eq $p[1] -and $_.method -eq $key })
        if ($r.Count -ne 1) { throw "Missing or duplicate row $task/$key" }
        $cells += Fmt $r[0].mean $r[0].std 100
    }
    Add ('| ' + ($cells -join ' | ') + ' |')
}
Add ''
Add '| Baseline | 本次实现的含义 |'
Add '|---|---|'
Add '| tePU | 仅用目标域 PU 数据训练 |'
Add '| trPU | 仅用源域 PU 数据训练 |'
Add '| ftPU | 源域预训练，再用目标域 PU 数据微调 |'
Add '| mtPU | 共享特征、源/目标两个分类头，组合两域 PU 风险 |'
Add '| mtsPU | 单个共享分类器，组合源/目标 PU 风险 |'
Add '| DAPU | 两域 PU 风险 + MMD 特征对齐 |'
Add '| UDAPU | 源域 PU 风险 + 目标域无标签 MMD 对齐 |'
Add '| GIW | 先构造伪 PN 标签，再做动态重要性加权；本项目为近似实现 |'
Add ''
Add 'FoodStamp 是附录任务（地理区域偏移），修正重跑结果如下：'
Add ''
Add '| 方法 | FoodStamp Acc (%) |'
Add '|---|---:|'
foreach ($key in $methods.Keys) { $r = $food | Where-Object method -eq $key; Add ('| ' + $methods[$key] + ' | ' + (Fmt $r.mean $r.std 100) + ' |') }
Add ''
Source 'DIW 图像任务主表' "$diw/main_table.csv"
Source 'Diabetes 修正主表' "$diw/tabular_no_target_rescale_v2/main_table.csv"
Source 'FoodStamp 修正表' "$diw/tabular_no_target_rescale_v2/appendix_foodstamp.csv"
Source 'baseline 定义' "$diw/../README.md"
Add '## 2. DIW 与传统 two-step：修正版正式配对实验'
Add ''
Add '420 个结果、210 个配对单元；每任务为 (10P,50U)、(20P,100U)、(40P,200U) × seeds 0–9。每格均值 ± 样本标准差。two-step 先训练完整比率模型，再冻结逐样本权重训练分类器；使用修正后的每域总 batch=256。旧 two-step 数值已被这一批次替代。'
Add ''
$pairs = @(Import-Csv "$diw/diw_vs_two_step_paper_v1_final/paired_cells.csv")
if ($pairs.Count -ne 210 -or @($pairs | Group-Object task,target_pos,target_unl,seed | Where-Object Count -ne 1).Count -ne 0) { throw 'Pair coverage mismatch' }
Add '| 任务 | DIW / IWPU | two-step baseline | DIW − two-step (pp) |'
Add '|---|---:|---:|---:|'
foreach ($g in ($pairs | Group-Object task | Sort-Object Name)) {
    if ($g.Count -ne 30) { throw 'Task coverage mismatch' }
    $a = Stats $g.Group.iwpu 1; $b = Stats $g.Group.two_step 1
    Add ('| {0} | {1} | {2} | {3:+0.00;-0.00;0.00} |' -f $g.Name, (Fmt $a[0] $a[1] 100), (Fmt $b[0] $b[1] 100), (100*($a[0]-$b[0])))
}
$a = Stats $pairs.iwpu 1; $b = Stats $pairs.two_step 1
Add ('| 全部任务等权汇总（均值） | {0:F2} | {1:F2} | {2:+0.00;-0.00;0.00} |' -f (100*$a[0]), (100*$b[0]), (100*($a[0]-$b[0])))
Add ''
Add '本批次总体 DIW 高 2.87 个百分点；优势不覆盖全部任务（MNIST IO 中 two-step 更高）。属于独立 from-paper 实现，不能称为作者官方复现。'
Source '最终逐配对准确率' "$diw/diw_vs_two_step_paper_v1_final/paired_cells.csv"
Source '协议修正记录' "$diw/diw_two_step_reaudit.md"
Add '## 3. SCAR：原始域数据 baseline 对照'
Add ''
Add 'MNIST、MLP、目标域固定旋转 30°；源域弱标注数据 59,000，目标域弱标注数据 1,000，独立目标域测试集 10,000。200 epochs，seeds 0–4。每个种子取最后 10 个 epoch 的目标分类准确率均值，再报告 5 个种子的均值 ± 总体标准差。'
Add ''
Add '| 方法 | 含义 | Acc (%) |'
Add '|---|---|---:|'
$defs = @{'trainSCAR'='仅源域弱标注数据';'testSCAR'='仅目标域弱标注数据';'ftSCAR / pooled SCAR'='直接混合源域和目标域，不使用 ADIW';'ADIW-SCAR'='两域数据 + ADIW 权重';'Conditioned-Ratio-SCAR'='条件化神经比率估计器';'Multihead-Ratio-SCAR'='多头神经比率估计器'}
$sc = @(Import-Csv "$scar/outputs/scarce_comparison/scar_method_comparison.csv")
foreach ($name in @('trainSCAR','testSCAR','ftSCAR / pooled SCAR','ADIW-SCAR','Conditioned-Ratio-SCAR','Multihead-Ratio-SCAR')) {
    $r = $sc | Where-Object method -eq $name
    $s = Stats ($r.seed_accuracies.Split(';'))
    if ([Math]::Abs($s[0]-[double]$r.mean_accuracy) -gt 0.00001 -or [Math]::Abs($s[1]-[double]$r.std_accuracy) -gt 0.00001) { throw 'SCAR aggregate mismatch' }
    Add ('| ' + $name + ' | ' + $defs[$name] + ' | ' + (Fmt $s[0] $s[1]) + ' |')
}
Add ''
Add '注意：本仓库的 ftSCAR 是 pooled SCAR 的别名，不能按 ftPU 的含义解释成“先预训练后微调”。ADIW 相对 pooled 的均值增益仅 0.67 pp，原报告配对 p=0.417，尚不能称为明确提升。'
Source 'SCAR 原始 baseline 及逐种子结果' "$scar/outputs/scarce_comparison/original_ABLATION_REPORT.md"
Source '含神经比率方法的准确率表' "$scar/outputs/scarce_comparison/scar_method_comparison.csv"
Add '## 4. SCAR：后续独立批次（补充，勿与第 3 节混表）'
Add ''
Add 'Phase A 在统一 runner 中比较 no-IW、pooled、ADIW 与神经比率方法。下面每格为 5 种子均值 ± 总体标准差，仍使用最后 10 个 epoch 的目标准确率。'
Add ''
$pa = @(Import-Csv "$scar/outputs/wisteria_next_round_phase_a/summary/summary_by_setting.csv")
Add '| 方法 | 旋转 10° | 旋转 20° | 旋转 30° |'
Add '|---|---:|---:|---:|'
foreach ($method in @('no_iw','pooled_scar','adiw','unified','multihead','fusion','separate_full')) {
    $cells = @($method)
    foreach ($angle in @(10,20,30)) { $r = @($pa | Where-Object { $_.method -eq $method -and $_.setting.StartsWith("rotation_${angle}_") }); if ($r.Count -eq 1) { $cells += Fmt $r[0].mean_target_accuracy $r[0].population_sd_target_accuracy } else { $cells += '未列入该批次' } }
    Add ('| ' + ($cells -join ' | ') + ' |')
}
Add ''
Source 'Phase A 分设置统计' "$scar/outputs/wisteria_next_round_phase_a/summary/summary_by_setting.csv"
Add 'Phase B0 的训练流程复核：MNIST 旋转 30°，10 种子，均值 ± 总体标准差。'
Add ''
Add '| 方法 / 干预 | Acc (%) |'
Add '|---|---:|'
$pb = @(Get-ChildItem "$scar/outputs/phase_b/runs/B0" -Recurse -Filter summary.json | ForEach-Object { Get-Content -LiteralPath $_.FullName -Raw | ConvertFrom-Json -AsHashtable })
foreach ($g in ($pb | Group-Object { $_['method'] } | Sort-Object Name)) { if ($g.Count -ne 10) { throw 'B0 coverage mismatch' }; $s = Stats @($g.Group | ForEach-Object { $_['last10_target_accuracy'] }); Add ('| ' + $g.Name + ' | ' + (Fmt $s[0] $s[1]) + ' |') }
Add ''
Add 'current_no_iw 是当前未匹配流程；matched_no_iw 匹配 ADIW 的训练流程但不使用 IW；adiw_ones 将权重置 1；adiw_mean 用均值权重；adiw_shuffle 打乱样本与权重对应关系；adiw_original 使用原始权重。该批次表明训练流程差异会明显影响 baseline，不能仅依据跨批次 acc 判断重要性加权的收益。'
Source 'Phase B0 报告' "$scar/reports/phase_b0_adiw_causal_decomposition.md"
Add '## 5. 核对范围与使用说明'
Add ''
Add '- 已检查 DIW 主表 63 个方法×任务条目，每项 30 次；Diabetes/FoodStamp 使用预处理修正版。'
Add '- 已从最终配对 CSV 独立重算 DIW/two-step 均值和标准差，核对 210 个唯一配对键、每任务 30 个。'
Add '- 已从 SCAR 逐种子数据独立重算第 3 节均值与总体标准差；B0 每方法 10 次，数值由正式 runs/B0 下逐次 summary.json 重算。'
Add '- DIW 的 std 为样本标准差；SCAR 历史报告使用总体标准差，两者均已明确标注。没有将域判别器 accuracy 或 ratio RMSE 当作分类 acc。'
Add '- 本汇总聚焦 baseline 准确率；神经架构额外批次、四类偏移压力测试、Gaussian oracle、B2 clipping 网格和 B4 异质偏移保留在原始报告中，不拼成一个排行榜。'
Add '- 置信评估：可附版本与口径说明转发。历史 DIW 主表并非最终 batch 协议下全方法重跑，不支持跨版本的严格优劣结论。'
Add ''
Source '本汇总的可重跑提取与核对脚本' "$scar/reports/build_baseline_summary.ps1"
$lines | Set-Content -LiteralPath $out -Encoding utf8
Write-Output "Saved $out ($($lines.Count) lines). Validation passed."
