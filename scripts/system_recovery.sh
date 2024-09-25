#!/bin/bash

# Check if the user is root
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root." 
   exit 1
fi

# Set source (the backup file) and destination (the system disk)
DEST_DISK="/dev/nvme0n1p1"      # Change this to the disk you want to restore to (ensure it's the correct disk)

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

GZ_BACKUP_FILE = "${FILE_DIR}/../system_backup_${USERNAME}.img.gz"
gzip -c -d "$GZ_BACKUP_FILE" > "$BACKUP_FILE"

BACKUP_FILE="${FILE_DIR}/../system_backup_${USERNAME}.img"  # Specify the path where you want to save the backup

echo "Starting system recovery from $BACKUP_FILE to $DEST_DISK..."

# Perform the restore using dd
dd if=$BACKUP_FILE of=$DEST_DISK bs=64K status=progress conv=noerror,sync

# Check if dd was successful
if [ $? -eq 0 ]; then
    echo "System recovery completed successfully."
else
    echo "System recovery failed."
fi
