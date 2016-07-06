#!/bin/bash
#
# simply displays the current GPU usage and memory consumption every
# second on the console, or in genmon bar style when started with
# --genmon
#
# works only on Nvidia cards
#

export GPU=0
export MEM=0

function genmon_mode
{
    get_gpu
    get_mem

    if [ ${GPU} -lt 10 ]; then
        GPU=0${GPU}
    fi

    if [ ${MEM} -lt 10 ]; then
        MEM=0${MEM}
    fi

    echo "<img>/usr/share/pixmaps/nvidia-current-settings.png</img>"
    echo "<txt><b><big><tt>${GPU} ${MEM}</tt></big></b></txt>"
    echo "<tool>GPU: ${GPU}%, Memory: ${MEM}%</tool>"
}

function console_mode
{
    echo -e "\033[30;42mPress Ctrl-C to break ...\033[0m"

    while true; do
        get_gpu
        get_mem
        #get_free_mem
        get_temp

        ## gpu usage
        echo -ne "\033[5G"
        echo -ne "\033[1;37;41mGPU:\033[0m ${GPU} "

        ## gpu memory
        #echo -ne "\033[15G"
        #echo -ne "\033[1;37;41mMEM:\033[0m ${MEM}   |"

        ## gpu free memory
        #echo -ne "\033[28G"
        #echo -ne "\033[0m ${FMEM}  | ${PMEM} %"

        echo -ne "\033[15G"
        echo -ne "\033[1;37;41mMEM:\033[0m u: ${MEM} | f: ${FMEM}  | ${PMEM} %"

        ## gpu temperature
        echo -ne "\033[52G"
        echo -e "\033[1;37;41mTEMP:\033[0m ${TEMP} "

        sleep 3
    done
}

function get_mem
{
    MEM=`optirun nvidia-smi -q --display=MEMORY 1>&1| grep -A 2 -i Used | head -1 | awk '{print $3,$4}' | sed s/\%//g`
    # free memory
    FMEM=`optirun nvidia-smi -q --display=MEMORY 1>&1| grep -A 2 -i Free | head -1 | awk '{print $3,$4}' | sed s/\%//g`
    # total memory number
    TMEM=`optirun nvidia-smi -q --display=MEMORY 1>&1| grep -A 2 -i Total | head -1 | awk '{print $3}' | sed s/\%//g`
    # used
    UMEM=`optirun nvidia-smi -q --display=MEMORY 1>&1| grep -A 2 -i Used | head -1 | awk '{print $3}' | sed s/\%//g`
    #percentage
    PMEM=$(echo $UMEM*100/$TMEM | bc)
}

function get_temp
{
    TEMP=`optirun nvidia-smi -q --display=TEMPERATURE 1>&1| grep -A 2 -i GPU\ Current\ Temp | head -1 | awk '{print $5,$6}' | sed s/\%//g`
}

function get_gpu
{
    GPU=`optirun nvidia-smi -q 1>&1| grep -A 2 -i Attached\ GPUs | head -1 | awk '{print $4}' | sed s/\%//g`
}

if [ $# -eq 0 ]; then
    console_mode
else
    case $1 in
        "--genmon") genmon_mode;;
        "--console") console_mode;;
    esac
fi
