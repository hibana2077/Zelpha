#!/bin/bash
# Submit all MBK parameter sweep jobs

echo "Submitting 36 MBK experiments..."
echo ""

qsub MBK001.sh
qsub MBK002.sh
qsub MBK003.sh
qsub MBK004.sh
qsub MBK005.sh
qsub MBK006.sh
qsub MBK007.sh
qsub MBK008.sh
qsub MBK009.sh
qsub MBK010.sh
qsub MBK011.sh
qsub MBK012.sh
qsub MBK013.sh
qsub MBK014.sh
qsub MBK015.sh
qsub MBK016.sh
qsub MBK017.sh
qsub MBK018.sh
qsub MBK019.sh
qsub MBK020.sh
qsub MBK021.sh
qsub MBK022.sh
qsub MBK023.sh
qsub MBK024.sh
qsub MBK025.sh
qsub MBK026.sh
qsub MBK027.sh
qsub MBK028.sh
qsub MBK029.sh
qsub MBK030.sh
qsub MBK031.sh
qsub MBK032.sh
qsub MBK033.sh
qsub MBK034.sh
qsub MBK035.sh
qsub MBK036.sh

echo ""
echo "All jobs submitted!"
echo "Check status with: qstat -u $USER"
