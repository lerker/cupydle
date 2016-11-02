# contar la cantidad de espacios por linea de un archivo
awk -F'|' 'BEGIN{print "count", "lineNum"}{print gsub(/ /,"") "\t" NR}' miarchivo.csv

# gsub() function's return value is number of substitution made. So we use that to print the number.
# NR holds the line number so we use it to print the line number.
# For printing occurrences of particular field, we create a variable fld and put the field number we wish to extract counts from.

grep -n -o "t" miarchivo.csv | sort -n | uniq -c | cut -d : -f 1
