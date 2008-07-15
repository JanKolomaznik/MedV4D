#ifdef _MSC_VER
#  pragma once
#endif

/**
 *  Bitmap fonts for glBitmap().
 *  License: General Public License GPL (http://www.gnu.org/copyleft/gpl.html)
 *  @author  Josef Pelikan $Author: pepca $
 *  @version $Rev: 16 $
 *  @date    $Date: 2005-04-16 03:11:01 +0200 (Sat, 16 Apr 2005) $
 */
#ifndef _FONTS_H
#define _FONTS_H

#include "base.h"
#include "vector.h"

//--- bitmap fonts drawing -------------------------------

/**
 *  Sets raster coordinates for drawText.
 *  Calls glPushMatrix() for GL_MODELVIEW and GL_PROJECTION.
 *  unsetTextCoords() can restore the matrices back..
 */
void setTextCoords ( int width, int height );

/**
 *  Sets actual text coordinates.
 *  Should be called inside setTextCoords() and unsetTextCoords().
 */
void setTextPosition ( int x, int y );

/**
 *  Reverts original coordinates after drawing text.
 */
void unsetTextCoords ();

/**
 *  Draws the given string at position set by setTextPosition().
 *  Can handle '\n' line delimiters.
 */
void drawText ( const char *str, int fontNumber =0 );

//--- bitmap fonts data ----------------------------------

/// "Filled" space character (for debugging purposes).
//#define SPACE_FILLED

/// Bitmap font #0: 8x16.
extern unsigned8 fontA8x16[];

/// Bitmap font #1: 8x16.
extern unsigned8 fontB8x16[];

#endif
