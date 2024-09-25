#!/bin/bash

# Check if the user is root
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root." 
   exit 1
fi

FILE_DIR=$(cd $(dirname $BASH_SOURCE) && pwd)

# Extract the home folder and onwards if the path is under /home
if [[ $FILE_DIR == /home/* ]]; then
    # Extract only the username from the path
    USERNAME=$(echo "$FILE_DIR" | sed -e 's|/home/\([^/]*\).*|\1|')
    echo "Username: $USERNAME"
else
    echo "Executable is not in the /home directory."
    exit 1
fi

BACKUP_FILE="${FILE_DIR}/../system_backup_${USERNAME}.tar.gz"
SOURCE_DIRECTORIES="/ /boot /home"

# Exclude files that shouldn't be backed up (e.g., /proc, /sys, /dev)
EXCLUDES="--exclude=/proc --exclude=/sys --exclude=/dev --exclude=/run --exclude=/mnt --exclude=/media --exclude=/tmp --exclude=/swapfile"

echo "Starting system backup from $SOURCE_DIRECTORIES to $BACKUP_FILE..."
tar czp --one-file-system $EXCLUDES -f $BACKUP_FILE $SOURCE_DIRECTORIES

# Check if tar was successful
if [ $? -eq 0 ]; then
  echo "Backup completed successfully. Archive saved to $BACKUP_FILE."
else
  echo "Backup failed."
fi