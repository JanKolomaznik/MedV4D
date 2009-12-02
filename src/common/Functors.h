#ifndef _FUNCTORS_H
#define _FUNCTORS_H

/**
 *  @ingroup common
 *  @file Functors.h
 *
 *  @addtogroup common
 *  @{
 *  @section functors Functors
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

template< typename TypeName >
struct TypeBox
{
	typedef TypeName type;
};

struct MakeTypeBox
{
	template< typename TypeName >
	struct apply {
		typedef TypeBox< TypeName > type;
	};
};

}/*namespace Functors*/
}/*namespace M4D*/

/** @} */

#endif /*_FUNCTORS_H*/

