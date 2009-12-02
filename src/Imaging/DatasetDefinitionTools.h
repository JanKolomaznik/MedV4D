/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file ADataset.h 
 * @{ 
 **/

#ifndef _DATA_SET_DEFINITION_TOOLS_H
#define _DATA_SET_DEFINITION_TOOLS_H


#include <boost/shared_ptr.hpp>
#include "common/Common.h"
/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

namespace M4D
{
namespace Imaging
{

#define MANDATORY_ROOT_DATASET_DEFINITIONS_MACRO( ... ) \
	typedef __VA_ARGS__				ThisClass; \
	typedef boost::shared_ptr< __VA_ARGS__ >	Ptr; \
	static const unsigned HierarchyDepth = 2;

#define MANDATORY_DATASET_DEFINITIONS_THIS_MACRO( ... ) \
	typedef __VA_ARGS__				ThisClass; \
	typedef boost::shared_ptr< __VA_ARGS__ >	Ptr; \
	
#define MANDATORY_DATASET_DEFINITIONS_PREDEC_MACRO( ... ) \
	typedef __VA_ARGS__	PredecessorType; \
	static const unsigned HierarchyDepth = __VA_ARGS__::HierarchyDepth + 1;
	
#define PREPARE_CAST_REFERENCE_MACRO	\
	static ThisClass & \
	Cast( ADataset & dataset ) \
	{	try { return dynamic_cast< ThisClass & >( dataset ); } \
		catch( ... ) {	_THROW_ ErrorHandling::ExceptionCastProblem();	} \
	} 

#define PREPARE_CAST_CONST_REFERENCE_MACRO	\
	static const ThisClass & \
	Cast( const ADataset & dataset ) \
	{	try { return dynamic_cast< const ThisClass & >( dataset ); } \
		catch( ... ) {	_THROW_ ErrorHandling::ExceptionCastProblem();	} \
	} 

#define PREPARE_CAST_SMART_POINTER_MACRO	\
	static Ptr \
	Cast( ADataset::Ptr dataset ) \
	{	if( dynamic_cast< ThisClass * >( dataset.get() ) == NULL ) { \
			_THROW_ ErrorHandling::ExceptionCastProblem();\
		} \
		return boost::static_pointer_cast< ThisClass >( dataset ); \
	}
	
#define PREPARE_CAST_METHODS_MACRO	\
	PREPARE_CAST_REFERENCE_MACRO \
	PREPARE_CAST_CONST_REFERENCE_MACRO \
	PREPARE_CAST_SMART_POINTER_MACRO

#define IS_CONSTRUCTABLE_MACRO \
	static const bool IsConstructable = true;

#define IS_NOT_CONSTRUCTABLE_MACRO \
	static const bool IsConstructable = false;




}/*namespace Imaging*/
}/*namespace M4D*/

/** @} */

#endif /*_DATA_SET_DEFINITION_TOOLS_H*/


/** @} */


