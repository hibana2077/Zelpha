#!/bin/bash
# Submit all RVGMBK parameter sweep jobs

echo "Submitting 36 RVGMBK experiments..."
echo ""

qsub RVGMBK001.sh
qsub RVGMBK002.sh
qsub RVGMBK003.sh
qsub RVGMBK004.sh
qsub RVGMBK005.sh
qsub RVGMBK006.sh
qsub RVGMBK007.sh
qsub RVGMBK008.sh
qsub RVGMBK009.sh
qsub RVGMBK010.sh
qsub RVGMBK011.sh
qsub RVGMBK012.sh
qsub RVGMBK013.sh
qsub RVGMBK014.sh
qsub RVGMBK015.sh
qsub RVGMBK016.sh
qsub RVGMBK017.sh
qsub RVGMBK018.sh
qsub RVGMBK019.sh
qsub RVGMBK020.sh
qsub RVGMBK021.sh
qsub RVGMBK022.sh
qsub RVGMBK023.sh
qsub RVGMBK024.sh
qsub RVGMBK025.sh
qsub RVGMBK026.sh
qsub RVGMBK027.sh
qsub RVGMBK028.sh
qsub RVGMBK029.sh
qsub RVGMBK030.sh
qsub RVGMBK031.sh
qsub RVGMBK032.sh
qsub RVGMBK033.sh
qsub RVGMBK034.sh
qsub RVGMBK035.sh
qsub RVGMBK036.sh

echo ""
echo "All jobs submitted!"
echo "Check status with: qstat -u $USER"
