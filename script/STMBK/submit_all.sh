#!/bin/bash
# Submit all STMBK parameter sweep jobs

echo "Submitting 36 STMBK experiments..."
echo ""

qsub STMBK001.sh
qsub STMBK002.sh
qsub STMBK003.sh
qsub STMBK004.sh
qsub STMBK005.sh
qsub STMBK006.sh
qsub STMBK007.sh
qsub STMBK008.sh
qsub STMBK009.sh
qsub STMBK010.sh
qsub STMBK011.sh
qsub STMBK012.sh
qsub STMBK013.sh
qsub STMBK014.sh
qsub STMBK015.sh
qsub STMBK016.sh
qsub STMBK017.sh
qsub STMBK018.sh
qsub STMBK019.sh
qsub STMBK020.sh
qsub STMBK021.sh
qsub STMBK022.sh
qsub STMBK023.sh
qsub STMBK024.sh
qsub STMBK025.sh
qsub STMBK026.sh
qsub STMBK027.sh
qsub STMBK028.sh
qsub STMBK029.sh
qsub STMBK030.sh
qsub STMBK031.sh
qsub STMBK032.sh
qsub STMBK033.sh
qsub STMBK034.sh
qsub STMBK035.sh
qsub STMBK036.sh

echo ""
echo "All jobs submitted!"
echo "Check status with: qstat -u $USER"
