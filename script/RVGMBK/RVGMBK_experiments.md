# RVGMBK Parameter Sweep Experiments

## Overview
This document tracks all RVGMBK parameter sweep experiments.

**Total Experiments:** 36

**Parameter Ranges:**
- `num_prototypes`: {1, 3, 5, 8}
- `beta`: {0.01, 0.1, 1.0}
- `margin`: {0.5, 1.0, 2.0}
- `linear_epochs`: 50
- `finetune_epochs`: 30

## Experiment Table

| Exp ID | num_prototypes | beta | margin | linear_epochs | finetune_epochs | robust_acc |
|--------|----------------|------|--------|---------------|-----------------|---|
| RVGMBK001 | 1 | 0.01 | 0.5 | 50 | 50 | 0.5262 |
| RVGMBK002 | 1 | 0.01 | 1.0 | 50 | 50 | 0.7762 |
| RVGMBK003 | 1 | 0.01 | 2.0 | 50 | 50 | 0.6262 |
| RVGMBK004 | 1 | 0.1 | 0.5 | 50 | 50 | 0.7857 |
| RVGMBK005 | 1 | 0.1 | 1.0 | 50 | 50 | 0.6571 |
| RVGMBK006 | 1 | 0.1 | 2.0 | 50 | 50 | 0.7048 |
| RVGMBK007 | 1 | 1.0 | 0.5 | 50 | 50 | 0.7857 |
| RVGMBK008 | 1 | 1.0 | 1.0 | 50 | 50 | 0.6881 |
| RVGMBK009 | 1 | 1.0 | 2.0 | 50 | 50 | 0.7500 |
| RVGMBK010 | 3 | 0.01 | 0.5 | 50 | 50 | 0.6643 |
| RVGMBK011 | 3 | 0.01 | 1.0 | 50 | 50 | 0.6286 |
| RVGMBK012 | 3 | 0.01 | 2.0 | 50 | 50 | 0.6738 |
| RVGMBK013 | 3 | 0.1 | 0.5 | 50 | 50 | 0.7595 |
| RVGMBK014 | 3 | 0.1 | 1.0 | 50 | 50 | 0.7595 |
| RVGMBK015 | 3 | 0.1 | 2.0 | 50 | 50 | 0.7381 |
| RVGMBK016 | 3 | 1.0 | 0.5 | 50 | 50 | 0.5167 |
| RVGMBK017 | 3 | 1.0 | 1.0 | 50 | 50 | 0.6548 |
| RVGMBK018 | 3 | 1.0 | 2.0 | 50 | 50 | 0.6667 |
| RVGMBK019 | 5 | 0.01 | 0.5 | 50 | 50 | 0.6405 |
| RVGMBK020 | 5 | 0.01 | 1.0 | 50 | 50 | 0.2857 |
| RVGMBK021 | 5 | 0.01 | 2.0 | 50 | 50 | 0.4429 |
| RVGMBK022 | 5 | 0.1 | 0.5 | 50 | 50 | 0.5976 |
| RVGMBK023 | 5 | 0.1 | 1.0 | 50 | 50 | 0.6000 |
| RVGMBK024 | 5 | 0.1 | 2.0 | 50 | 50 | 0.7071 |
| RVGMBK025 | 5 | 1.0 | 0.5 | 50 | 50 | 0.7048 |
| RVGMBK026 | 5 | 1.0 | 1.0 | 50 | 50 | 0.6119 |
| RVGMBK027 | 5 | 1.0 | 2.0 | 50 | 50 | 0.6857 |
| RVGMBK028 | 8 | 0.01 | 0.5 | 50 | 50 | 0.7476 |
| RVGMBK029 | 8 | 0.01 | 1.0 | 50 | 50 | 0.4595 |
| RVGMBK030 | 8 | 0.01 | 2.0 | 50 | 50 | 0.7429 |
| RVGMBK031 | 8 | 0.1 | 0.5 | 50 | 50 | 0.5190 |
| RVGMBK032 | 8 | 0.1 | 1.0 | 50 | 50 | 0.7476 |
| RVGMBK033 | 8 | 0.1 | 2.0 | 50 | 50 | 0.6810 |
| RVGMBK034 | 8 | 1.0 | 0.5 | 50 | 50 | 0.5333 |
| RVGMBK035 | 8 | 1.0 | 1.0 | 50 | 50 | 0.7833 |
| RVGMBK036 | 8 | 1.0 | 2.0 | 50 | 50 | 0.5929 |


## Notes
Update the Status and Results columns as experiments complete.

---
*Generated on: 2025-11-30*
