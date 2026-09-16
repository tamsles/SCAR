$ErrorActionPreference = 'Stop'
$root = 'C:/Users/Tamsles/Documents/SCAR'
$old = 'C:/Users/Tamsles/Documents/Codex/2026-07-10/wwangwitsel-scarce-git-https-github-com/work/SCARCE'
$shift = 'C:/Users/Tamsles/Documents/Codex/2026-07-14/rang/outputs/wisteria_shift_ablation_20260715'
$doc = [System.Collections.Generic.List[string]]::new()
$checks = [System.Collections.Generic.List[string]]::new()
function A([string]$s = '') { $doc.Add($s) }
function Src([string]$name, [string]$path) { A ''; A "Source: [$name](<$path>)."; A '' }
function Stat($values) {
    $v = @($values | ForEach-Object { if ($null -eq $_ -or "$_" -eq '') { throw 'Missing accuracy' }; [double]$_ })
    if (!$v.Count) { throw 'Empty group' }
    $mean = ($v | Measure-Object -Average).Average
    $variance = ($v | ForEach-Object { [Math]::Pow($_ - $mean, 2) } | Measure-Object -Sum).Sum / $v.Count
    return @($mean, [Math]::Sqrt($variance))
}
function F($mean, $sd, [int]$digits=2) { return ('{0:F' + $digits + '} ± {1:F' + $digits + '}') -f [double]$mean,[double]$sd }
function Cell($rows, [int]$n) {
    $rr = @($rows)
    if ($rr.Count -ne $n) { throw "Expected $n records; got $($rr.Count)" }
    if (@($rr | Group-Object { $_['seed'] } | Where-Object Count -ne 1).Count) { throw 'Duplicate seeds' }
    $s = Stat @($rr | ForEach-Object { $_['last10_target_accuracy'] })
    return F $s[0] $s[1]
}
$pb = @{}
foreach ($phase in @('B0','B1','B2','B3','B4')) {
    $pb[$phase] = @(Get-ChildItem "$root/outputs/phase_b/runs/$phase" -Recurse -Filter summary.json | ForEach-Object { Get-Content $_.FullName -Raw | ConvertFrom-Json -AsHashtable })
    $expected = @{B0=60;B1=20;B2=57;B3=400;B4=100}[$phase]
    if ($pb[$phase].Count -ne $expected) { throw "Wrong count: $phase" }
    foreach ($r in $pb[$phase]) { if ($r['status'] -ne 'ok' -or $r['target_test_used_for_selection'] -ne $false) { throw "Invalid record in $phase" } }
    $checks.Add("$phase`: $expected records, successful status, no target-test selection.")
}
A '# SCAR / Complementary-Label Learning: Complete Available Experiment Inventory'
A ''
A 'Prepared for supervisor review, September 8, 2026. This document consolidates all formal SCAR/CLL experiment batches located in the local repositories and synchronized results, plus the early incomplete SCARCE run. It does not substitute PU-learning results for CLL results.'
A ''
A '**Reporting convention.** Classification accuracy is in percent. Unless explicitly stated otherwise, each run contributes its mean target accuracy over the final 10 classifier epochs, and tables report the mean ± population standard deviation across seeds. Different batches remain separate because their data splits, training paths, or ratio settings differ. Smoke/debug runs are not treated as completed scientific comparisons.'
A ''
A '**Fine-tuning versus pooling.** Both are listed explicitly. Genuine fine-tuning means source pretraining followed by target-domain fine-tuning; no completed SCAR result for that procedure was located. Pooled SCAR trains on combined source/target data without importance weighting. The historical label ftSCAR refers to this pooled implementation, not a separate fine-tuning experiment.'
A ''
A '## 1. Method inventory and missing baseline results'
A ''
A '| Method or family | What is available |'
A '|---|---|'
A '| Original SCARCE | One completed MNIST run and one incomplete run; Section 2. |'
A '| Source-only SCAR / trainSCAR / no_iw / current_no_iw | Completed results in Sections 3, 6, 7, and 11; these names occur in different training batches. |'
A '| Target-only SCAR / testSCAR | Completed original five-seed comparison; Section 3. |'
A '| Pooled SCAR (historically labeled ftSCAR) | Completed original comparison, four-shift experiment, and Phase A results; Sections 3, 4, and 6. Trains on combined source and target data without importance weighting. |'
A '| ADIW-SCAR / adiw / adiw_original | Completed domain-shift, Phase A, B0, B1, and B4 experiments. Uses online KMM weights in this implementation. |'
A '| Conditioned / unified, multihead, fusion, separate | Completed neural density-ratio comparisons; Sections 3, 5, and 6, with further diagnostics in Phase B. |'
A '| matched_no_iw, adiw_ones, adiw_mean, adiw_shuffle | Completed training-pipeline and weight-assignment interventions; Section 7. |'
A '| Estimated global/conditional ratios and oracle ratios | Completed Gaussian CLL benchmark; Section 10. Oracle access is diagnostic only. |'
A '| mtSCAR (source/target multi-task classifiers) | No completed SCAR training result or dedicated implementation was located in the inspected local repositories. Earlier discussion proposed the CLL analogue of mtPU; no verified SCAR accuracy is available. |'
A '| mtsSCAR (single-network multi-task SCAR) | No separately identified completed result or dedicated implementation was located. It must not be assigned mtsPU accuracy or silently equated with pooled SCAR. |'
A '| Genuine source-pretraining/target-fine-tuning SCAR | No completed result was located. The historical ftSCAR label did not implement this procedure. |'
A '| DANN/MMD/CLARINET + SCAR; pseudo-label GIW + SCAR | Discussed as possible CLL baselines; no completed results were located in the inspected SCAR outputs. PU DAPU/UDAPU/GIW results are not SCAR results. |'
A '| DIW mode in the early SCARCE main.py | A code path exists, but no separate completed DIW-mode result was located in the early result directory. Do not equate this with the later ADIW-SCAR result. |'
A ''
A 'The earlier baseline-design discussion mapped mtPU and mtsPU to potential CLL counterparts. That discussion is evidence of a proposal, not a completed experiment. The absence statements above are limited to the local code/results and task history inspected; they are not proof that no unsynchronized external run exists.'
A ''
A '## 2. Early original SCARCE run (incomplete repetition set)'
A ''
$earlyPath = "$old/result/detail/Res_detail_mnist_random_SCARCE_mlp_adam_lr0.001_wd1e-05_b256_e200_s0_r5.csv"
$early = @(Import-Csv $earlyPath)
$r1 = @($early | Where-Object run_idx -eq 1)
$r2 = @($early | Where-Object run_idx -eq 2)
if ($r1.Count -ne 200 -or $r2.Count -ne 94) { throw 'Legacy run counts changed; review needed' }
$last = Stat @($r1 | Select-Object -Last 10 -ExpandProperty test_accuracy)
if ([Math]::Abs($last[0]-91.052) -gt 0.00001) { throw 'Legacy mean mismatch' }
A '| Experiment | Available completion | Accuracy (%) |'
A '|---|---|---:|'
A '| Original SCARCE, MNIST, MLP, 200 epochs | Run 1 complete; reported last-10-epoch mean | 91.052 |'
A '| Same intended five-run experiment | Run 2 saved through epoch 94; complete five-run aggregate absent | Not available |'
A ''
A 'This is the original MNIST SCARCE setting, not the later source/target rotation-30 protocol. The single completed result cannot be reported as a five-seed mean ± SD or compared directly with the shifted-domain tables.'
Src 'early per-epoch record' $earlyPath
A '## 3. Original domain-training and neural-ratio comparison'
A ''
A 'MNIST + MLP; target rotation 30°; source weak set 59,000; target weak training set 1,000; independent target test set 10,000. Adam, learning rate 0.001, weight decay 0.00001, batch size 256, 200 classifier epochs; five seeds (0–4). Ordinary labels are used only for evaluation.'
A ''
A '| Method | Accuracy (%) |'
A '|---|---:|'
$sc = @(Import-Csv "$root/outputs/scarce_comparison/scar_method_comparison.csv")
foreach ($r in $sc) {
    $s = Stat $r.seed_accuracies.Split(';')
    if ([Math]::Abs($s[0]-[double]$r.mean_accuracy) -gt 0.00001 -or [Math]::Abs($s[1]-[double]$r.std_accuracy) -gt 0.00001) { throw 'Original SCAR mismatch' }
    A "| $($r.method) | $(F $s[0] $s[1]) |"
}
A '| Fine-tuning SCAR (source pretraining → target fine-tuning) | Unavailable: no completed result located |'
A ''
A 'trainSCAR uses only source complementary labels. testSCAR uses only the small target weak training set, never the held-out test set. Conditioned-Ratio-SCAR conditions a shared estimator on class and weak-label state; Multihead-Ratio-SCAR uses shared features with separate conditional-ratio outputs. Their mean accuracies are close to ADIW-SCAR; this batch does not establish a neural-ratio accuracy advantage.'
Src 'original comparison and per-seed values' "$root/outputs/scarce_comparison/scar_method_comparison.csv"
A '## 4. Four types of distribution shift'
A ''
A 'Separate MNIST/MLP batch; 59,000 source and 1,000 target weak examples, 200 epochs, five seeds. Both ADIW and the unweighted pooled-data baseline are included.'
A ''
A '| Shift intervention | Method | Accuracy (%) |'
A '|---|---|---:|'
$shiftNames = @('Covariate shift: target rotation 30°','Label-conditional shift: target labels cyclically permuted by +1','Complementary-label mechanism shift','Joint shift: all three interventions')
$shiftRows = @(Import-Csv "$shift/shift_ablation_summary.csv")
for ($i=0;$i -lt $shiftRows.Count;$i++) {
    $r=$shiftRows[$i]; $s=Stat $r.adiw_seed_accuracies.Split(';')
    if ([Math]::Abs($s[0]-[double]$r.adiw_mean) -gt 0.00001) { throw 'Shift mean mismatch' }
    A "| $($shiftNames[$i]) | ADIW-SCAR | $(F $s[0] $s[1]) |"
    $p=Stat $r.pooled_seed_accuracies.Split(';')
    if ([Math]::Abs($p[0]-[double]$r.pooled_mean) -gt 0.00001 -or [Math]::Abs($p[1]-[double]$r.pooled_std) -gt 0.00001) { throw 'Pooled shift mismatch' }
    A "| $($shiftNames[$i]) | Pooled SCAR (no IW) | $(F $p[0] $p[1]) |"
}
A ''
A 'The very low accuracy under label permutation belongs to a deliberate label-mapping stress test. It is not the standard rotation-only benchmark.'
Src 'four-shift aggregate and per-seed results' "$shift/shift_ablation_summary.csv"
A '## 5. Full neural-ratio architecture comparison'
A ''
A 'Separate 20-run batch: four architectures × five seeds, MNIST rotation 30°, 10 ratio-training epochs and 200 classifier epochs. The classifier and data sizes match the stated MNIST family, but these values must remain separate from Section 3.'
A ''
A '| Architecture | Description | Accuracy (%) |'
A '|---|---|---:|'
$desc = @{unified='Shared class/state-conditioned ratio estimator';multihead='Shared backbone with branch-specific ratio heads';fusion='Learned combination of global and conditional ratios';separate='Independent ratio estimators for the class/state branches'}
$full = Get-Content "$root/outputs/wisteria_full_results_20260725/summary.json" -Raw | ConvertFrom-Json -AsHashtable
foreach ($m in @('unified','multihead','fusion','separate')) { $r=$full.metrics[$m]; $s=Stat $r.target_accuracy_by_seed; if ([Math]::Abs($s[0]-$r.target_accuracy_mean) -gt 0.00001) { throw 'Architecture mismatch' }; A "| $m | $($desc[$m]) | $(F $s[0] $s[1]) |" }
Src 'full architecture results' "$root/outputs/wisteria_full_results_20260725/summary.json"
A '## 6. Phase A: rotation severity and matched architecture comparison'
A ''
A 'The full batch contains 95 runs, including the pooled comparator. Each available method/angle cell contains five seeds. MNIST source size 59,000, target weak size 1,000; ratio clipping [0.01, 20] and conditional-branch normalization. The no_iw baseline is source-only; it does not match the ADIW source/target mixing pipeline.'
A ''
$pa = @(Import-Csv "$root/outputs/wisteria_next_round_phase_a/summary/summary_by_setting.csv")
if (($pa | Measure-Object num_seeds -Sum).Sum -ne 95) { throw 'Phase A count mismatch' }
A '| Method | Rotation 10° | Rotation 20° | Rotation 30° |'
A '|---|---:|---:|---:|'
foreach ($m in @('no_iw','pooled_scar','adiw','unified','multihead','fusion','separate_full')) {
    $cells=@($m)
    foreach ($angle in @(10,20,30)) {
        $r=@($pa | Where-Object { $_.method -eq $m -and $_.setting.StartsWith("rotation_${angle}_") })
        if ($r.Count -eq 0) { $cells+='Not run in this matrix' } else { $s=Stat $r[0].seed_accuracies.Split(';'); if ([Math]::Abs($s[0]-[double]$r[0].mean_target_accuracy) -gt 0.00001) { throw 'Phase A mismatch' }; $cells+=F $s[0] $s[1] }
    }
    A ('| '+($cells -join ' | ')+' |')
}
Src 'Phase A setting summary' "$root/outputs/wisteria_next_round_phase_a/summary/summary_by_setting.csv"
A '## 7. Phase B0: weight assignment and training-pipeline interventions'
A ''
A '60 completed runs: six interventions × ten seeds. MNIST rotation 30°, target weak size 1,000. Source-only and pipeline-matched unit-weight baselines are explicitly distinguished.'
A ''
$b0desc=@{current_no_iw='Current source-only no-IW baseline';matched_no_iw='ADIW-matched source/target mixing and schedule, without a ratio estimator';adiw_original='Original online KMM importance weights';adiw_ones='ADIW path with all source weights replaced by one';adiw_mean='ADIW path with sample weights replaced by their mean';adiw_shuffle='ADIW path with weights shuffled across samples'}
A '| Method | Intervention | Accuracy (%) |'
A '|---|---|---:|'
foreach ($m in @('current_no_iw','matched_no_iw','adiw_original','adiw_ones','adiw_mean','adiw_shuffle')) { $r=@($pb.B0 | Where-Object {$_['method'] -eq $m}); A "| $m | $($b0desc[$m]) | $(Cell $r 10) |" }
A ''
A 'The gain from matching the training pipeline is much larger than the difference between original and unit/shuffled weights. A source-only baseline alone cannot isolate the benefit of importance weighting.'
Src 'B0 report' "$root/reports/phase_b0_adiw_causal_decomposition.md"
A '## 8. Phase B1: ratio correctness audit (no CLL classification accuracy)'
A ''
A '20 completed audits: four methods × five seeds. These runs do not train/evaluate a downstream CLL classifier. Their domain-discriminator accuracy must not be reported as target class accuracy. Lower weighted MMD and domain AUC closer to 0.5 indicate better domain matching.'
A ''
A '| Method | CLL accuracy | Mean weighted MMD | Mean post-hoc domain AUC | Mean domain accuracy (%) |'
A '|---|---|---:|---:|---:|'
foreach ($g in ($pb.B1 | Group-Object {$_['method']} | Sort-Object Name)) {
    if ($g.Count -ne 5) { throw 'B1 count mismatch' }
    $mmd=Stat @($g.Group | ForEach-Object {$_['weighted_MMD']}); $auc=Stat @($g.Group | ForEach-Object {$_['posthoc_domain_AUC']}); $acc=Stat @($g.Group | ForEach-Object {$_['posthoc_domain_accuracy']})
    A ('| {0} | Not applicable | {1:F4} | {2:F4} | {3:F2} |' -f $g.Name,$mmd[0],$auc[0],(100*$acc[0]))
}
Src 'B1 audit report' "$root/reports/phase_b1_ratio_audit.md"
A '## 9. Phase B2: frozen-score clipping, normalization, and calibration'
A ''
A '57 classifier runs: 15 uncalibrated settings × three seeds, plus four calibrated settings × three seeds. A single unified ratio estimator is trained per seed; frozen scores are reused. MNIST rotation 30°, upper ratio clip 20, 200 classifier epochs. Calibration candidates were selected without target class labels.'
A ''
A '| Lower ratio clip | Normalization | Calibration | Accuracy (%) |'
A '|---:|---|---|---:|'
foreach ($lower in @(0,0.000001,0.0001,0.001,0.01)) {
    foreach ($norm in @('none','global','conditional_branch')) {
        foreach ($cal in @('none','temperature','platt')) {
            $r=@($pb.B2 | Where-Object { $_['lower_clip'] -eq $lower -and $_['normalization'] -eq $norm -and $_['calibration'] -eq $cal })
            if ($r.Count) { A "| $lower | $norm | $cal | $(Cell $r 3) |" }
        }
    }
}
A ''
A 'Higher lower-clipping floors strongly improve accuracy in this grid. This should not be interpreted as successful density-ratio recovery: the largest improvements occur when many raw ratios are collapsed to the floor. The grid is a diagnostic comparison, not a test-label-selected final model.'
Src 'B2 report' "$root/reports/phase_b2_postprocessing_ablation.md"
A '## 10. Phase B3: Gaussian CLL benchmark with known density ratios'
A ''
A '400 runs: four shifts × five methods × twenty seeds. Three classes, 2,000 source examples, 1,000 target weak examples, 2,000 target test examples; 50 classifier epochs. Oracle rows use analytically known ratios and are diagnostic references, not deployable weakly supervised baselines.'
A ''
A '| Method | No shift | Homogeneous conditional shift | Heterogeneous conditional shift | Class-prior shift only |'
A '|---|---:|---:|---:|---:|'
foreach ($m in @('no_iw','global_estimated_ratio','conditional_estimated_ratio','oracle_global_ratio','oracle_conditional_ratio')) {
    $cells=@($m)
    foreach ($setting in @('no_shift','homogeneous_conditional_shift','heterogeneous_conditional_shift','class_prior_shift_only')) { $r=@($pb.B3 | Where-Object {$_['method'] -eq $m -and $_['shift_type'] -eq $setting}); $cells+=Cell $r 20 }
    A ('| '+($cells -join ' | ')+' |')
}
Src 'B3 ground-truth benchmark report' "$root/reports/phase_b3_gaussian_ground_truth.md"
A '## 11. Phase B4: class-dependent MNIST rotations'
A ''
A '100 runs: five rotation patterns × four methods × five seeds. Source weak size 59,000, target weak size 1,000; 10 ratio epochs and 200 classifier epochs; clipping [0.01, 20] with conditional-branch normalization.'
A ''
A '| Setting | Target rotation angles for classes 0–9 |'
A '|---|---|'
A '| H0: no shift | 0° for every class |'
A '| H1: homogeneous | +30° for every class |'
A '| H2: heterogeneous magnitude | 10°, 10°, 20°, 20°, 30°, 30°, 40°, 40°, 50°, 50° |'
A '| H3: sparse shift | 0° for classes 0–4; +60° for classes 5–9 |'
A '| H4: opposite directions | +30° for classes 0–4; −30° for classes 5–9 |'
A ''
A '| Method | H0 | H1 | H2 | H3 | H4 |'
A '|---|---:|---:|---:|---:|---:|'
foreach ($m in @('current_no_iw','adiw_original','unified','multihead')) {
    $cells=@($m)
    foreach ($setting in @('H0_no_shift','H1_homogeneous','H2_magnitude_heterogeneous','H3_sparse_heterogeneous','H4_direction_heterogeneous')) { $r=@($pb.B4 | Where-Object {$_['method'] -eq $m -and $_['shift_type'] -eq $setting}); $cells+=Cell $r 5 }
    A ('| '+($cells -join ' | ')+' |')
}
A ''
A 'Neither conditional architecture met the preregistered practical-improvement criterion against ADIW in any heterogeneous setting (paired gain >1 percentage point with a positive lower 95% t-confidence bound).'
Src 'B4 report' "$root/reports/phase_b4_heterogeneous_mnist.md"
A '## 12. Earlier synthetic ratio-estimation ablations (no downstream CLL accuracy)'
A ''
A 'Two completed synthetic batches, each with ten methods and seeds 7, 17, 29. These test density-ratio recovery rather than classifier accuracy. The table reports log-ratio RMSE, mean ± population SD; lower is better. Domain-validation accuracy in these files is not CLL classification accuracy.'
A ''
$mixed=@(Import-Csv "$root/outputs/ablation_mixed/summary.csv")
$branch=@(Import-Csv "$root/outputs/ablation_branch_shift/summary.csv")
A '| Ratio estimator / ablation | Mixed covariate and mild label shift: RMSE | Heterogeneous branch shift: RMSE | CLL accuracy |'
A '|---|---:|---:|---|'
foreach ($r in $mixed) { $b=$branch | Where-Object name -eq $r.name; if (!$b) {throw 'Missing synthetic method'}; A "| $($r.name) | $(F $r.log_ratio_rmse_mean $r.log_ratio_rmse_std 4) | $(F $b.log_ratio_rmse_mean $b.log_ratio_rmse_std 4) | Not measured |" }
A ''
A 'proposed is the conditioned estimator with balanced BCE; branch_none removes branch balancing; branch_inverse_frequency uses inverse-frequency balancing; no_class_condition/no_state_condition remove the respective conditioning input; no_prior_correction omits count-prior correction in the relevant unbalanced setup; marginal_ratio estimates a single pooled density ratio; unit_ratio uses weights of one; oracle_ratio uses known ratios. The marginal density-ratio baseline is distinct from pooling source and target classifier training data.'
Src 'mixed synthetic ratio ablation' "$root/outputs/ablation_mixed/summary.csv"
Src 'heterogeneous synthetic ratio ablation' "$root/outputs/ablation_branch_shift/summary.csv"
A '## 13. Completeness and provenance'
A ''
A '- All five formal Phase B stages are included: B0=60, B1=20, B2=57, B3=400, B4=100 (637 records). Classification means and population SDs were recomputed from per-run summary.json files; group sizes and unique seeds were checked.'
A '- Original SCAR, four-shift, full-architecture, and Phase A means were checked against stored per-seed values. The 91.052% early SCARCE value was recomputed from the last ten epochs of its one completed run.'
A '- The initial five-seed ADIW result also appears in a separate July 14 archive; it is the same 60.978 ± 2.438% result in Section 3, not an additional independent experiment.'
A '- Incomplete runs and smoke/debug runs are not silently promoted to formal results. The incomplete original repetition set is explicitly shown in Section 2.'
A '- mtSCAR/mtsSCAR are explicitly listed as lacking verified results. Proposed baseline names and available PU implementations are not evidence that their CLL versions were run.'
A '- This is an inventory of located local/synchronized evidence. No new models were trained and no unsynchronized remote files were inspected for this compilation.'
A ''
Src 'reproducible aggregation and checks' "$root/reports/build_all_scar_en.ps1"
$doc | Set-Content "$root/reports/SCAR_CLL_all_experiments_EN.md" -Encoding utf8
$checks.Add('Original baseline, shift, architecture and Phase A aggregates independently checked.')
$checks.Add('Legacy SCARCE: run 1 has 200 epochs; run 2 has 94; run-1 last-10 mean is 91.052%.')
$checks | Set-Content "$root/reports/SCAR_CLL_all_experiments_validation.txt" -Encoding utf8
Write-Output "Saved English inventory: $($doc.Count) lines. All checks passed."
