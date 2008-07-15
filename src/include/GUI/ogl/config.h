#ifdef _MSC_VER
#  pragma once
#endif

/**
 * Configuration symbols. Has to be included before any other header!
 * License: General Public License GPL (http://www.gnu.org/copyleft/gpl.html)
 * @author  Josef Pelikan $Author: pepca $
 * @version $Rev: 53 $
 * @date    $Date: 2006-04-10 02:42:04 +0200 (Mon, 10 Apr 2006) $
 */
#ifndef _CONFIG_H
#define _CONFIG_H

// Uncomment following lines to see all warnings ("deprecated" etc.)
#pragma warning( disable : 4005 )
#pragma warning( disable : 4996 )

//--- system time ----------------------------------------

//#define HAS_CLOCK_GETTIME                   // not yet implemented in Linux ??
#define HAS_GETTIMEOFDAY

//--- text logging ---------------------------------------

#define USE_LOG
#include "GUI/ogl/logflags.h"

#ifndef REV_NO
#  define REV_NO(x)    (atoi((x)+6))
#  define CONFIG_REV   REV_NO("$Rev: 53 $")
#endif

//--- miscellaneous - compiler specific ------------------

#ifdef _MSC_VER

#  define CDECL   __cdecl
#  define MUTABLE mutable
#  define CONST   const
#  define COMPILE_ASSERT(cond)  { char d[ (cond) ? 1 : -1 ]; (void)d; }

#else     // _MSC_VER

#  define CDECL
#  define MUTABLE mutable
#  define CONST   const
#  define COMPILE_ASSERT(cond)  { char d[ (cond) ? 0 : -1 ]; (void)d; }

#endif    // _MSC_VER

#ifdef __GNUC__

#  define INIT_PRIORITY_NORMAL __attribute__ ((init_priority(10000)))
#  define INIT_PRIORITY_HIGH   __attribute__ ((init_priority(9000)))
#  define INIT_PRIORITY_URGENT __attribute__ ((init_priority(8000)))
#  define INIT_PRIORITY(prio)  __attribute__ ((init_priority(prio)))

#  define PACKED               __attribute__ ((packed))

#else     // __GNUC__

#  define INIT_PRIORITY_NORMAL
#  define INIT_PRIORITY_HIGH
#  define INIT_PRIORITY_URGENT
#  define INIT_PRIORITY(prio)

# define PACKED

#endif    // __GNUC__

//--- miscellaneous - platform specific ------------------

#ifdef _WIN32

#  ifndef PATH_MAX
#  define PATH_MAX  1024
#  endif

#else

#  ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#  endif
#  define PATH_MAX  1024

#endif    // _WIN32

#endif
