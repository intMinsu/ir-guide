#!/bin/bash

# Check if the user is root
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root." 
   exit 1
fi

# Set source (your system disk, e.g., /dev/sda) and destination (backup file)
SOURCE_DISK="/dev/nvme0n1p1"      # Change this to your system disk
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

BACKUP_FILE="${FILE_DIR}/../system_backup_${USERNAME}.img"  # Specify the path where you want to save the backup

echo "Starting system backup from $SOURCE_DISK to $BACKUP_FILE..."

# Perform the backup using dd
dd if=$SOURCE_DISK of=$BACKUP_FILE bs=64K status=progress conv=noerror,sync

# # Check if dd was successful
if [ $? -eq 0 ]; then
    echo "Backup completed successfully."
else
    echo "Backup failed."
fi

COMPRESSED_FILE="${BACKUP_FILE}.gz"

# Compress the image file using gzip (lossless)
echo "Compressing $BACKUP_FILE to $COMPRESSED_FILE..."

gzip -c "$BACKUP_FILE" > "$COMPRESSED_FILE"

# Check if gzip succeeded
if [ $? -eq 0 ]; then
    echo "Compression completed successfully: $COMPRESSED_FILE"
else
    echo "Compression failed."
    exit 1
fi
