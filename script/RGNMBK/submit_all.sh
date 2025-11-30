#!/bin/bash
# Submit all RGNMBK parameter sweep jobs

echo "Submitting 36 RGNMBK experiments..."
echo ""

qsub RGNMBK001.sh
qsub RGNMBK002.sh
qsub RGNMBK003.sh
qsub RGNMBK004.sh
qsub RGNMBK005.sh
qsub RGNMBK006.sh
qsub RGNMBK007.sh
qsub RGNMBK008.sh
qsub RGNMBK009.sh
qsub RGNMBK010.sh
qsub RGNMBK011.sh
qsub RGNMBK012.sh
qsub RGNMBK013.sh
qsub RGNMBK014.sh
qsub RGNMBK015.sh
qsub RGNMBK016.sh
qsub RGNMBK017.sh
qsub RGNMBK018.sh
qsub RGNMBK019.sh
qsub RGNMBK020.sh
qsub RGNMBK021.sh
qsub RGNMBK022.sh
qsub RGNMBK023.sh
qsub RGNMBK024.sh
qsub RGNMBK025.sh
qsub RGNMBK026.sh
qsub RGNMBK027.sh
qsub RGNMBK028.sh
qsub RGNMBK029.sh
qsub RGNMBK030.sh
qsub RGNMBK031.sh
qsub RGNMBK032.sh
qsub RGNMBK033.sh
qsub RGNMBK034.sh
qsub RGNMBK035.sh
qsub RGNMBK036.sh

echo ""
echo "All jobs submitted!"
echo "Check status with: qstat -u $USER"
