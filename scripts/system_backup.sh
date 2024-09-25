#!/bin/bash

# Check if the user is root
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root." 
   exit 1
fi

# Set source (your system disk, e.g., /dev/sda) and destination (backup file)
SOURCE_DISK="/dev/nvme0n1p1"      # Change this to your system disk
BACKUP_FILE="/home/$(whoami)/system_backup_$(whoami).img"  # Specify the path where you want to save the backup

echo "Starting system backup from $SOURCE_DISK to $BACKUP_FILE..."

# Perform the backup using dd
dd if=$SOURCE_DISK of=$BACKUP_FILE bs=64K status=progress conv=noerror,sync

# Check if dd was successful
if [ $? -eq 0 ]; then
    echo "Backup completed successfully."
else
    echo "Backup failed."
fi
