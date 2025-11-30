#!/bin/bash
# Submit all HL50MBK parameter sweep jobs

echo "Submitting 36 HL50MBK experiments..."
echo ""

qsub HL50MBK001.sh
qsub HL50MBK002.sh
qsub HL50MBK003.sh
qsub HL50MBK004.sh
qsub HL50MBK005.sh
qsub HL50MBK006.sh
qsub HL50MBK007.sh
qsub HL50MBK008.sh
qsub HL50MBK009.sh
qsub HL50MBK010.sh
qsub HL50MBK011.sh
qsub HL50MBK012.sh
qsub HL50MBK013.sh
qsub HL50MBK014.sh
qsub HL50MBK015.sh
qsub HL50MBK016.sh
qsub HL50MBK017.sh
qsub HL50MBK018.sh
qsub HL50MBK019.sh
qsub HL50MBK020.sh
qsub HL50MBK021.sh
qsub HL50MBK022.sh
qsub HL50MBK023.sh
qsub HL50MBK024.sh
qsub HL50MBK025.sh
qsub HL50MBK026.sh
qsub HL50MBK027.sh
qsub HL50MBK028.sh
qsub HL50MBK029.sh
qsub HL50MBK030.sh
qsub HL50MBK031.sh
qsub HL50MBK032.sh
qsub HL50MBK033.sh
qsub HL50MBK034.sh
qsub HL50MBK035.sh
qsub HL50MBK036.sh

echo ""
echo "All jobs submitted!"
echo "Check status with: qstat -u $USER"
