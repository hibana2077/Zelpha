#!/bin/bash
# Submit all HL26MBK parameter sweep jobs

echo "Submitting 36 HL26MBK experiments..."
echo ""

qsub HL26MBK001.sh
qsub HL26MBK002.sh
qsub HL26MBK003.sh
qsub HL26MBK004.sh
qsub HL26MBK005.sh
qsub HL26MBK006.sh
qsub HL26MBK007.sh
qsub HL26MBK008.sh
qsub HL26MBK009.sh
qsub HL26MBK010.sh
qsub HL26MBK011.sh
qsub HL26MBK012.sh
qsub HL26MBK013.sh
qsub HL26MBK014.sh
qsub HL26MBK015.sh
qsub HL26MBK016.sh
qsub HL26MBK017.sh
qsub HL26MBK018.sh
qsub HL26MBK019.sh
qsub HL26MBK020.sh
qsub HL26MBK021.sh
qsub HL26MBK022.sh
qsub HL26MBK023.sh
qsub HL26MBK024.sh
qsub HL26MBK025.sh
qsub HL26MBK026.sh
qsub HL26MBK027.sh
qsub HL26MBK028.sh
qsub HL26MBK029.sh
qsub HL26MBK030.sh
qsub HL26MBK031.sh
qsub HL26MBK032.sh
qsub HL26MBK033.sh
qsub HL26MBK034.sh
qsub HL26MBK035.sh
qsub HL26MBK036.sh

echo ""
echo "All jobs submitted!"
echo "Check status with: qstat -u $USER"
