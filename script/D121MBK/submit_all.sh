#!/bin/bash
# Submit all D121MBK parameter sweep jobs

echo "Submitting 36 D121MBK experiments..."
echo ""

qsub D121MBK001.sh
qsub D121MBK002.sh
qsub D121MBK003.sh
qsub D121MBK004.sh
qsub D121MBK005.sh
qsub D121MBK006.sh
qsub D121MBK007.sh
qsub D121MBK008.sh
qsub D121MBK009.sh
qsub D121MBK010.sh
qsub D121MBK011.sh
qsub D121MBK012.sh
qsub D121MBK013.sh
qsub D121MBK014.sh
qsub D121MBK015.sh
qsub D121MBK016.sh
qsub D121MBK017.sh
qsub D121MBK018.sh
qsub D121MBK019.sh
qsub D121MBK020.sh
qsub D121MBK021.sh
qsub D121MBK022.sh
qsub D121MBK023.sh
qsub D121MBK024.sh
qsub D121MBK025.sh
qsub D121MBK026.sh
qsub D121MBK027.sh
qsub D121MBK028.sh
qsub D121MBK029.sh
qsub D121MBK030.sh
qsub D121MBK031.sh
qsub D121MBK032.sh
qsub D121MBK033.sh
qsub D121MBK034.sh
qsub D121MBK035.sh
qsub D121MBK036.sh

echo ""
echo "All jobs submitted!"
echo "Check status with: qstat -u $USER"
