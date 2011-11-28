#ifndef TYPE_COMPARATOR_H
#define TYPE_COMPARATOR_H

#include "MedV4D/Common/Types.h"
#include <boost/numeric/conversion/conversion_traits.hpp>

template< typename FirstType, typename SecondType >
struct TypeComparator
{
	typedef typename boost::numeric::conversion_traits< FirstType, SecondType>::supertype 	Superior;
	typedef typename boost::numeric::conversion_traits< FirstType, SecondType>::subtype 	Inferior;
};



#endif /*TYPE_COMPARATOR_H*/
