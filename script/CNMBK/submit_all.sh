#!/bin/bash
# Submit all CNMBK parameter sweep jobs

echo "Submitting 36 CNMBK experiments..."
echo ""

qsub CNMBK001.sh
qsub CNMBK002.sh
qsub CNMBK003.sh
qsub CNMBK004.sh
qsub CNMBK005.sh
qsub CNMBK006.sh
qsub CNMBK007.sh
qsub CNMBK008.sh
qsub CNMBK009.sh
qsub CNMBK010.sh
qsub CNMBK011.sh
qsub CNMBK012.sh
qsub CNMBK013.sh
qsub CNMBK014.sh
qsub CNMBK015.sh
qsub CNMBK016.sh
qsub CNMBK017.sh
qsub CNMBK018.sh
qsub CNMBK019.sh
qsub CNMBK020.sh
qsub CNMBK021.sh
qsub CNMBK022.sh
qsub CNMBK023.sh
qsub CNMBK024.sh
qsub CNMBK025.sh
qsub CNMBK026.sh
qsub CNMBK027.sh
qsub CNMBK028.sh
qsub CNMBK029.sh
qsub CNMBK030.sh
qsub CNMBK031.sh
qsub CNMBK032.sh
qsub CNMBK033.sh
qsub CNMBK034.sh
qsub CNMBK035.sh
qsub CNMBK036.sh

echo ""
echo "All jobs submitted!"
echo "Check status with: qstat -u $USER"
