#! /bin/bash

cd /workdir/

if [ $# -eq 1 ]; then
    cd $1
fi

lst=(*/)
if [ ${#lst[@]} -eq 0 ]; then
    echo "No folders found in $PWD"
    exit 1
fi

if [ -z "${lst[$BATCH_TASK_INDEX]}" ]; then
    echo "No folders found in $PWD"
    exit 1
fi

cd ${lst[$BATCH_TASK_INDEX]}
echo "Entered $PWD"
bash run.sh