
============================================================
SWEEP COMPLETE — Results summary
============================================================
   r   alpha    α/r    micro_f1    macro_f1     hamming
------------------------------------------------------------
  16      32    2.0      0.6547      0.6212      0.1037
  32      64    2.0      0.6527      0.6187      0.1052
  16      16    1.0      0.6429      0.6032      0.1066
  32      32    1.0      0.6401      0.6024      0.1071
   8       8    1.0      0.6387      0.6012      0.1069
   8      16    2.0      0.6293      0.5806      0.1095

Best config → r=16, alpha=32  (micro_f1=0.6547)

  sweep_report.txt written  → /nfs/u50/laiv3/Methods Comparison/Embedding Matrix/lora_ann_runs/sweep_report.txt        

Output files in /nfs/u50/laiv3/Methods Comparison/Embedding Matrix/lora_ann_runs:
  sweep_results.csv   — one row per run
  epoch_losses.csv    — per-epoch train/val loss
  sweep_report.txt    — human-readable summary
  r*/loss_curve_*.png — loss curve per run



============================================================
SWEEP COMPLETE — Results summary
============================================================
    m   red_factor    micro_f1    macro_f1     hamming
------------------------------------------------------------
  256          3.0      0.7038      0.6846      0.0948

Best config → m=256  (micro_f1=0.7038)

  sweep_report.txt written  → /nfs/u50/laiv3/Methods Comparison/Embedding Matrix/bn_ann_runs/sweep_report.txt

Output files in /nfs/u50/laiv3/Methods Comparison/Embedding Matrix/bn_ann_runs:
  sweep_results.csv    — one row per run
  epoch_losses.csv     — per-epoch train/val loss
  sweep_report.txt     — human-readable summary
  m*/loss_curve_*.png  — loss curve per run