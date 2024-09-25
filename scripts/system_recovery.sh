#!/bin/bash

# Check if the user is root
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root." 
   exit 1
fi

# Set source (the backup file) and destination (the system disk)
BACKUP_FILE="/home/$(whoami)/system_backup_$(whoami).img"  # Path to the backup image
DEST_DISK="/dev/nvme0n1p1"      # Change this to the disk you want to restore to (ensure it's the correct disk)

echo "Starting system recovery from $BACKUP_FILE to $DEST_DISK..."

# Perform the restore using dd
dd if=$BACKUP_FILE of=$DEST_DISK bs=64K status=progress conv=noerror,sync

# Check if dd was successful
if [ $? -eq 0 ]; then
    echo "System recovery completed successfully."
else
    echo "System recovery failed."
fi
