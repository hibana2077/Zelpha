# MBCMBK Parameter Sweep Experiments

## Overview
This document tracks all MBCMBK parameter sweep experiments.

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
| MBCMBK001 | 1 | 0.01 | 0.5 | 50 | 50 | 0.2405 |
| MBCMBK002 | 1 | 0.01 | 1.0 | 50 | 50 | 0.1762 |
| MBCMBK003 | 1 | 0.01 | 2.0 | 50 | 50 | 0.2024 |
| MBCMBK004 | 1 | 0.1 | 0.5 | 50 | 50 | 0.1524 |
| MBCMBK005 | 1 | 0.1 | 1.0 | 50 | 50 | 0.1667 |
| MBCMBK006 | 1 | 0.1 | 2.0 | 50 | 50 | 0.2119 |
| MBCMBK007 | 1 | 1.0 | 0.5 | 50 | 50 | 0.2262 |
| MBCMBK008 | 1 | 1.0 | 1.0 | 50 | 50 | 0.2405 |
| MBCMBK009 | 1 | 1.0 | 2.0 | 50 | 50 | 0.1714 |
| MBCMBK010 | 3 | 0.01 | 0.5 | 50 | 50 | 0.1429 |
| MBCMBK011 | 3 | 0.01 | 1.0 | 50 | 50 | 0.1952 |
| MBCMBK012 | 3 | 0.01 | 2.0 | 50 | 50 | 0.2048 |
| MBCMBK013 | 3 | 0.1 | 0.5 | 50 | 50 | 0.3048 |
| MBCMBK014 | 3 | 0.1 | 1.0 | 50 | 50 | 0.3048 |
| MBCMBK015 | 3 | 0.1 | 2.0 | 50 | 50 | 0.3000 |
| MBCMBK016 | 3 | 1.0 | 0.5 | 50 | 50 | 0.1952 |
| MBCMBK017 | 3 | 1.0 | 1.0 | 50 | 50 | 0.2667 |
| MBCMBK018 | 3 | 1.0 | 2.0 | 50 | 50 | 0.1500 |
| MBCMBK019 | 5 | 0.01 | 0.5 | 50 | 50 | 0.2167 |
| MBCMBK020 | 5 | 0.01 | 1.0 | 50 | 50 | 0.1429 |
| MBCMBK021 | 5 | 0.01 | 2.0 | 50 | 50 | 0.1714 |
| MBCMBK022 | 5 | 0.1 | 0.5 | 50 | 50 | 0.2571 |
| MBCMBK023 | 5 | 0.1 | 1.0 | 50 | 50 | 0.2024 |
| MBCMBK024 | 5 | 0.1 | 2.0 | 50 | 50 | 0.1714 |
| MBCMBK025 | 5 | 1.0 | 0.5 | 50 | 50 | 0.1929 |
| MBCMBK026 | 5 | 1.0 | 1.0 | 50 | 50 | 0.1857 |
| MBCMBK027 | 5 | 1.0 | 2.0 | 50 | 50 | 0.1643 |
| MBCMBK028 | 8 | 0.01 | 0.5 | 50 | 50 | 0.2071 |
| MBCMBK029 | 8 | 0.01 | 1.0 | 50 | 50 | 0.1548 |
| MBCMBK030 | 8 | 0.01 | 2.0 | 50 | 50 | 0.2333 |
| MBCMBK031 | 8 | 0.1 | 0.5 | 50 | 50 | 0.1429 |
| MBCMBK032 | 8 | 0.1 | 1.0 | 50 | 50 | 0.2095 |
| MBCMBK033 | 8 | 0.1 | 2.0 | 50 | 50 | 0.2571 |
| MBCMBK034 | 8 | 1.0 | 0.5 | 50 | 50 | 0.2643 |
| MBCMBK035 | 8 | 1.0 | 1.0 | 50 | 50 | 0.2167 |
| MBCMBK036 | 8 | 1.0 | 2.0 | 50 | 50 | 0.1595 |


## Notes
Update the Status and Results columns as experiments complete.

---
*Generated on: 2025-11-30*