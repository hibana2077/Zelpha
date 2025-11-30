# RGNMBK Parameter Sweep Experiments

## Overview
This document tracks all RGNMBK parameter sweep experiments.

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
| RGNMBK001 | 1 | 0.01 | 0.5 | 50 | 50 | 0.6881 |
| RGNMBK002 | 1 | 0.01 | 1.0 | 50 | 50 | 0.6881 |
| RGNMBK003 | 1 | 0.01 | 2.0 | 50 | 50 | 0.6881 |
| RGNMBK004 | 1 | 0.1 | 0.5 | 50 | 50 | 0.6857 |
| RGNMBK005 | 1 | 0.1 | 1.0 | 50 | 50 | 0.6857 |
| RGNMBK006 | 1 | 0.1 | 2.0 | 50 | 50 | 0.6857 |
| RGNMBK007 | 1 | 1.0 | 0.5 | 50 | 50 | 0.6857 |
| RGNMBK008 | 1 | 1.0 | 1.0 | 50 | 50 | 0.6857 |
| RGNMBK009 | 1 | 1.0 | 2.0 | 50 | 50 | 0.6857 |
| RGNMBK010 | 3 | 0.01 | 0.5 | 50 | 50 | 0.6548 |
| RGNMBK011 | 3 | 0.01 | 1.0 | 50 | 50 | 0.6548 |
| RGNMBK012 | 3 | 0.01 | 2.0 | 50 | 50 | 0.6548 |
| RGNMBK013 | 3 | 0.1 | 0.5 | 50 | 50 | 0.6571 |
| RGNMBK014 | 3 | 0.1 | 1.0 | 50 | 50 | 0.6571 |
| RGNMBK015 | 3 | 0.1 | 2.0 | 50 | 50 | 0.6571 |
| RGNMBK016 | 3 | 1.0 | 0.5 | 50 | 50 | 0.4262 |
| RGNMBK017 | 3 | 1.0 | 1.0 | 50 | 50 | 0.6571 |
| RGNMBK018 | 3 | 1.0 | 2.0 | 50 | 50 | 0.6571 |
| RGNMBK019 | 5 | 0.01 | 0.5 | 50 | 50 | 0.5524 |
| RGNMBK020 | 5 | 0.01 | 1.0 | 50 | 50 | 0.5524 |
| RGNMBK021 | 5 | 0.01 | 2.0 | 50 | 50 | 0.5524 |
| RGNMBK022 | 5 | 0.1 | 0.5 | 50 | 50 | 0.5500 |
| RGNMBK023 | 5 | 0.1 | 1.0 | 50 | 50 | 0.5500 |
| RGNMBK024 | 5 | 0.1 | 2.0 | 50 | 50 | 0.5500 |
| RGNMBK025 | 5 | 1.0 | 0.5 | 50 | 50 | 0.6000 |
| RGNMBK026 | 5 | 1.0 | 1.0 | 50 | 50 | 0.6000 |
| RGNMBK027 | 5 | 1.0 | 2.0 | 50 | 50 | 0.6000 |
| RGNMBK028 | 8 | 0.01 | 0.5 | 50 | 50 | 0.6905 |
| RGNMBK029 | 8 | 0.01 | 1.0 | 50 | 50 | 0.6905 |
| RGNMBK030 | 8 | 0.01 | 2.0 | 50 | 50 | 0.6905 |
| RGNMBK031 | 8 | 0.1 | 0.5 | 50 | 50 | 0.6571 |
| RGNMBK032 | 8 | 0.1 | 1.0 | 50 | 50 | 0.6571 |
| RGNMBK033 | 8 | 0.1 | 2.0 | 50 | 50 | 0.6571 |
| RGNMBK034 | 8 | 1.0 | 0.5 | 50 | 50 | 0.6643 |
| RGNMBK035 | 8 | 1.0 | 1.0 | 50 | 50 | 0.6643 |
| RGNMBK036 | 8 | 1.0 | 2.0 | 50 | 50 | 0.6643 |


## Notes
Update the Status and Results columns as experiments complete.

---
*Generated on: 2025-11-30*
