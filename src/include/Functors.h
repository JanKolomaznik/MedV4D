#ifndef _FUNCTORS_H
#define _FUNCTORS_H

/**
 *  @ingroup common
 *  @file Functors.h
 *
 *  @addtogroup common
 *  @{
 *  @section Functors
 *
 *  destruction tools
 */

namespace M4D
{
namespace Functors
{

template< typename Pointer >
struct Deletor
{
	void
	operator()( Pointer& p )
	{ delete p; }
};

}/*namespace Functors*/
}/*namespace M4D*/

/** @} */

#endif /*_FUNCTORS_H*/

