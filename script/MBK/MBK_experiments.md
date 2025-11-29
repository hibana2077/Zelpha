# MBK Parameter Sweep Experiments

## Overview

This document tracks all MBK parameter sweep experiments.

**Total Experiments:** 36

**Parameter Ranges:**

- `num_prototypes`: {1, 3, 5, 8}
- `beta`: {0.01, 0.1, 1.0}
- `margin`: {0.5, 1.0, 2.0}
- `linear_epochs`: 50
- `finetune_epochs`: 30

## Experiment Table

| Exp ID | num_prototypes | beta | margin | linear_epochs | finetune_epochs | SC1.0 | SC1.15 | SC1.3 | SC0.7 | SC0.85|
|--------|----------------|------|--------|---------------|-----------------|--------|---|---|---|---|
| MBK001 | 1 | 0.01 | 0.5 | 50 | 30 | TBD | TBD | TBD | TBD | TBD |
| MBK002 | 1 | 0.01 | 1.0 | 50 | 30 | TBD | TBD | TBD | TBD | TBD |
| MBK003 | 1 | 0.01 | 2.0 | 50 | 30 | TBD | TBD | TBD | TBD | TBD |
| MBK004 | 1 | 0.1 | 0.5 | 50 | 30 | TBD | TBD | TBD | TBD | TBD |
| MBK005 | 1 | 0.1 | 1.0 | 50 | 30 | TBD | TBD | TBD | TBD | TBD |
| MBK006 | 1 | 0.1 | 2.0 | 50 | 30 | TBD | TBD | TBD | TBD | TBD |
| MBK007 | 1 | 1.0 | 0.5 | 50 | 30 | TBD | TBD | TBD | TBD | TBD |
| MBK008 | 1 | 1.0 | 1.0 | 50 | 30 | TBD | TBD | TBD | TBD | TBD |
| MBK009 | 1 | 1.0 | 2.0 | 50 | 30 | TBD | TBD | TBD | TBD | TBD |
| MBK010 | 3 | 0.01 | 0.5 | 50 | 30 | TBD | TBD | TBD | TBD | TBD |
| MBK011 | 3 | 0.01 | 1.0 | 50 | 30 | TBD | TBD | TBD | TBD | TBD |
| MBK012 | 3 | 0.01 | 2.0 | 50 | 30 | TBD | TBD | TBD | TBD | TBD |
| MBK013 | 3 | 0.1 | 0.5 | 50 | 30 | TBD | TBD | TBD | TBD | TBD |
| MBK014 | 3 | 0.1 | 1.0 | 50 | 30 | TBD | TBD | TBD | TBD | TBD |
| MBK015 | 3 | 0.1 | 2.0 | 50 | 30 | TBD | TBD | TBD | TBD | TBD |
| MBK016 | 3 | 1.0 | 0.5 | 50 | 30 | TBD | TBD | TBD | TBD | TBD |
| MBK017 | 3 | 1.0 | 1.0 | 50 | 30 | TBD | TBD | TBD | TBD | TBD |
| MBK018 | 3 | 1.0 | 2.0 | 50 | 30 | TBD | TBD | TBD | TBD | TBD |
| MBK019 | 5 | 0.01 | 0.5 | 50 | 30 | TBD | TBD | TBD | TBD | TBD |
| MBK020 | 5 | 0.01 | 1.0 | 50 | 30 | TBD | TBD | TBD | TBD | TBD |
| MBK021 | 5 | 0.01 | 2.0 | 50 | 30 | TBD | TBD | TBD | TBD | TBD |
| MBK022 | 5 | 0.1 | 0.5 | 50 | 30 | TBD | TBD | TBD | TBD | TBD |
| MBK023 | 5 | 0.1 | 1.0 | 50 | 30 | TBD | TBD | TBD | TBD | TBD |
| MBK024 | 5 | 0.1 | 2.0 | 50 | 30 | TBD | TBD | TBD | TBD | TBD |
| MBK025 | 5 | 1.0 | 0.5 | 50 | 30 | TBD | TBD | TBD | TBD | TBD |
| MBK026 | 5 | 1.0 | 1.0 | 50 | 30 | TBD | TBD | TBD | TBD | TBD |
| MBK027 | 5 | 1.0 | 2.0 | 50 | 30 | TBD | TBD | TBD | TBD | TBD |
| MBK028 | 8 | 0.01 | 0.5 | 50 | 30 | TBD | TBD | TBD | TBD | TBD |
| MBK029 | 8 | 0.01 | 1.0 | 50 | 30 | TBD | TBD | TBD | TBD | TBD |
| MBK030 | 8 | 0.01 | 2.0 | 50 | 30 | TBD | TBD | TBD | TBD | TBD |
| MBK031 | 8 | 0.1 | 0.5 | 50 | 30 | TBD | TBD | TBD | TBD | TBD |
| MBK032 | 8 | 0.1 | 1.0 | 50 | 30 | TBD | TBD | TBD | TBD | TBD |
| MBK033 | 8 | 0.1 | 2.0 | 50 | 30 | TBD | TBD | TBD | TBD | TBD |
| MBK034 | 8 | 1.0 | 0.5 | 50 | 30 | TBD | TBD | TBD | TBD | TBD |
| MBK035 | 8 | 1.0 | 1.0 | 50 | 30 | TBD | TBD | TBD | TBD | TBD |
| MBK036 | 8 | 1.0 | 2.0 | 50 | 30 | TBD | TBD | TBD | TBD | TBD |

## Notes

Update the Status and Results columns as experiments complete.

---
*Generated on: 2025-11-29*
