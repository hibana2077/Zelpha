#!/bin/bash
# Submit all DMBK parameter sweep jobs

echo "Submitting 36 DMBK experiments..."
echo ""

qsub DMBK001.sh
qsub DMBK002.sh
qsub DMBK003.sh
qsub DMBK004.sh
qsub DMBK005.sh
qsub DMBK006.sh
qsub DMBK007.sh
qsub DMBK008.sh
qsub DMBK009.sh
qsub DMBK010.sh
qsub DMBK011.sh
qsub DMBK012.sh
qsub DMBK013.sh
qsub DMBK014.sh
qsub DMBK015.sh
qsub DMBK016.sh
qsub DMBK017.sh
qsub DMBK018.sh
qsub DMBK019.sh
qsub DMBK020.sh
qsub DMBK021.sh
qsub DMBK022.sh
qsub DMBK023.sh
qsub DMBK024.sh
qsub DMBK025.sh
qsub DMBK026.sh
qsub DMBK027.sh
qsub DMBK028.sh
qsub DMBK029.sh
qsub DMBK030.sh
qsub DMBK031.sh
qsub DMBK032.sh
qsub DMBK033.sh
qsub DMBK034.sh
qsub DMBK035.sh
qsub DMBK036.sh

echo ""
echo "All jobs submitted!"
echo "Check status with: qstat -u $USER"
