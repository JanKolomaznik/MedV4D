#ifndef TYPE_COMPARATOR_H
#define TYPE_COMPARATOR_H

#include "common/Types.h"


template< typename FirstType, typename SecondType >
struct TypeComparator;

template< typename Type >
struct TypeComparator< Type, Type >
{
	typedef Type	Superior;
	typedef Type	Inferior;
};

//TODO - finish for all types


#endif /*TYPE_COMPARATOR_H*/
