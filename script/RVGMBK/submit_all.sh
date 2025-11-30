#!/bin/bash
# Submit all MBCMBK parameter sweep jobs

echo "Submitting 36 MBCMBK experiments..."
echo ""

qsub MBCMBK001.sh
qsub MBCMBK002.sh
qsub MBCMBK003.sh
qsub MBCMBK004.sh
qsub MBCMBK005.sh
qsub MBCMBK006.sh
qsub MBCMBK007.sh
qsub MBCMBK008.sh
qsub MBCMBK009.sh
qsub MBCMBK010.sh
qsub MBCMBK011.sh
qsub MBCMBK012.sh
qsub MBCMBK013.sh
qsub MBCMBK014.sh
qsub MBCMBK015.sh
qsub MBCMBK016.sh
qsub MBCMBK017.sh
qsub MBCMBK018.sh
qsub MBCMBK019.sh
qsub MBCMBK020.sh
qsub MBCMBK021.sh
qsub MBCMBK022.sh
qsub MBCMBK023.sh
qsub MBCMBK024.sh
qsub MBCMBK025.sh
qsub MBCMBK026.sh
qsub MBCMBK027.sh
qsub MBCMBK028.sh
qsub MBCMBK029.sh
qsub MBCMBK030.sh
qsub MBCMBK031.sh
qsub MBCMBK032.sh
qsub MBCMBK033.sh
qsub MBCMBK034.sh
qsub MBCMBK035.sh
qsub MBCMBK036.sh

echo ""
echo "All jobs submitted!"
echo "Check status with: qstat -u $USER"
