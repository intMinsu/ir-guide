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

RESTORE_DESTINATION="/"
BACKUP_FILE="${FILE_DIR}/../system_backup_${USERNAME}.tar.gz"

echo "Starting tar recovery from $BACKUP_FILE..."
sudo tar xzp -f $BACKUP_FILE -C $RESTORE_DESTINATION

# Check if tar was successful
if [ $? -eq 0 ]; then
  echo "System recovery completed successfully."
else
  echo "Recovery failed."
fi