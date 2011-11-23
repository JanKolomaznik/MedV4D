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

#include <functional>


namespace M4D
{
namespace Functors
{

template< typename TPointer >
struct Deletor: public std::unary_function< TPointer, void >
{
	void
	operator()( TPointer& p )
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

struct PredicateAlwaysTrue
{
	typedef bool result_type;
	template < typename TType >
	bool
	operator()( TType p )
	{ return true; }
};

struct PredicateAlwaysFalse
{
	typedef bool result_type;
	template < typename TType >
	bool
	operator()( TType p )
	{ return false; }
};


}//namespace Functors
}//namespace M4D

/** @} */

#endif /*_FUNCTORS_H*/

