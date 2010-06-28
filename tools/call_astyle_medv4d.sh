#!/bin/sh 

OPTIONS="-y
--indent-blocks
--indent-brackets
--pad=oper
--pad=paren
--indent=spaces=8
--indent=tab=8
--force-indent=tab=8
--brackets=break
--mode=c"


if [ ! -n "$1" ]; then
echo "Syntax is: recurse.sh dirname filesuffix"
echo "Syntax is: recurse.sh filename"
echo "Example: recurse.sh temp cpp"
exit 1
fi

if [ -d "$1" ]; then
#echo "Dir ${1} exists"
if [ -n "$2" ]; then
filesuffix=$2
else
filesuffix="*"
fi

#echo "Filtering files using suffix ${filesuffix}"

file_list=`find ${1} -name "*.${filesuffix}" -type f`
for file2indent in $file_list
do 
echo "Indenting file $file2indent"

#!/bin/bash
astyle "$file2indent" $OPTIONS

done
else
if [ -f "$1" ]; then
echo "Indenting one file $1"
#!/bin/bash
astyle "$1" --options="/usr/share/universalindentgui/indenters/.astylerc"

else
echo "ERROR: As parameter given directory or file does not exist!"
echo "Syntax is: call_Artistic_Style.sh dirname filesuffix"
echo "Syntax is: call_Artistic_Style.sh filename"
echo "Example: call_Artistic_Style.sh temp cpp"
exit 1
fi
fi
