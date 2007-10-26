#!/bin/sh
#
# $Date: 2007-05-08 19:11:49 +0200 (út, 08 V 2007) $
# $Author: pepca $
# $Rev: 450 $
#
# Updates all MedV4D WWW data from SVN server.
#
# Usage: update.sh

P=/home/www/MedV4D

cd $P
date '+%d.%m.%y - %H:%M' >> www-update.err

echo =================
date '+%d.%m.%y - %H:%M'
echo -----------------
svn update
