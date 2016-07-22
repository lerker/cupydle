#!/bin/bash

SECONDS=0

if [ $# -eq 0 ]
then
    echo "debes introduccir el archivo .py a ejecutar"
    exit
fi

if [ ! $# -eq 1 ]
then
    echo "solo acepto un nombre de archivo a la vez"
    exit
fi

LOG_FILE="logFile"

# Close STDOUT file descriptor
exec 1<&-
# Close STDERR FD
exec 2<&-

# Open STDOUT as $LOG_FILE file for read and write.
exec 1<>$LOG_FILE

# Redirect STDERR to STDOUT
exec 2>&1

######
RED="\e[31m"
GREEN="\e[92m"
DEF="\e[39m"
BLUE="\e[34m"

echo ""
echo -e "      LOG FILE START TIME: " ${GREEN}$(date '+%Y-%m-%d    %H:%M:%S')
echo -e "${BLUE}============================================================${DEF}"


python3 $1

echo -e "${BLUE}============================================================${DEF}"
echo -e "      LOG FILE END TIME: " ${GREEN}$(date '+%Y-%m-%d    %H:%M:%S')

echo ""
duration=$SECONDS
echo -e "${DEF}It tooks ${RED}$(($duration / 1440 )) ${DEF}hours, ${RED}$(($duration / 60)) ${DEF}minutes and ${RED}$(($duration % 60)) ${DEF}seconds to finish."
echo ""

# $0 - The name of the Bash script.
# $1 - $9 - The first 9 arguments to the Bash script. (As mentioned above.)
# $# - How many arguments were passed to the Bash script.
# $@ - All the arguments supplied to the Bash script.
# $? - The exit status of the most recently run process.
# $$ - The process ID of the current script.
# $USER - The username of the user running the script.
# $HOSTNAME - The hostname of the machine the script is running on.
# $SECONDS - The number of seconds since the script was started.
# $RANDOM - Returns a different random number each time is it referred to.
# $LINENO - Returns the current line number in the Bash script.
