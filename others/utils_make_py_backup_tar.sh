#!/bin/bash

backup_archive="../all_py_backups"
mkdir ${backup_archive}

py_files=`find .|grep "\.py$"`
echo $py_files


for py_file in $py_files
do
    cp $py_file ${backup_archive}
done

