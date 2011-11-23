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

#define STANDARD_DECLARATIONS_MACRO( ... ) \
	typedef __VA_ARGS__				ThisClass; \
	typedef const ThisClass				ConstThisClass; \
	typedef boost::shared_ptr< __VA_ARGS__ >	Ptr; \
	typedef boost::shared_ptr< const __VA_ARGS__ >	ConstPtr; \
	typedef ThisClass &				Reference;\
	typedef const ThisClass &			ConstReference;\
	typedef ThisClass *				SimplePointer;\
	typedef const ThisClass *			ConstSimplePointer;

#define MANDATORY_ROOT_DATASET_DEFINITIONS_MACRO( ... ) \
	STANDARD_DECLARATIONS_MACRO( __VA_ARGS__ ) \
	static const unsigned HierarchyDepth = 2;

#define MANDATORY_DATASET_DEFINITIONS_THIS_MACRO( ... ) \
	STANDARD_DECLARATIONS_MACRO( __VA_ARGS__ ) \
	
	
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
#define PREPARE_CAST_CONST_SMART_POINTER_MACRO	\
	static ConstPtr \
	Cast( ADataset::ConstPtr dataset ) \
	{	if( dynamic_cast< const ThisClass * >( dataset.get() ) == NULL ) { \
			_THROW_ ErrorHandling::ExceptionCastProblem();\
		} \
		return boost::static_pointer_cast< const ThisClass >( dataset ); \
	}
	
#define PREPARE_CAST_METHODS_MACRO	\
	PREPARE_CAST_REFERENCE_MACRO \
	PREPARE_CAST_CONST_REFERENCE_MACRO \
	PREPARE_CAST_SMART_POINTER_MACRO \
	PREPARE_CAST_CONST_SMART_POINTER_MACRO

#define IS_CONSTRUCTABLE_MACRO \
	static const bool IsConstructable = true

#define IS_NOT_CONSTRUCTABLE_MACRO \
	static const bool IsConstructable = false






#define CONFIGURABLE_PREPARE_CAST_REFERENCE_MACRO( METHOD_NAME, RETURN_TYPE, INPUT_TYPE )	\
	static RETURN_TYPE::Reference \
	METHOD_NAME( INPUT_TYPE::Reference input ) \
	{	try { return dynamic_cast< RETURN_TYPE::Reference >( input ); } \
		catch( ... ) {	_THROW_ ErrorHandling::ExceptionCastProblem();	} \
	} 

#define CONFIGURABLE_PREPARE_CAST_CONST_REFERENCE_MACRO( METHOD_NAME, RETURN_TYPE, INPUT_TYPE )	\
	static RETURN_TYPE::ConstReference \
	METHOD_NAME( INPUT_TYPE::ConstReference input ) \
	{	try { return dynamic_cast< RETURN_TYPE::ConstReference >( input ); } \
		catch( ... ) {	_THROW_ ErrorHandling::ExceptionCastProblem();	} \
	} 

#define CONFIGURABLE_PREPARE_CAST_SMART_POINTER_MACRO( METHOD_NAME, RETURN_TYPE, INPUT_TYPE )	\
	static RETURN_TYPE::Ptr \
	METHOD_NAME( INPUT_TYPE::Ptr input ) \
	{	if( dynamic_cast< RETURN_TYPE::SimplePointer >( input.get() ) == NULL ) { \
			_THROW_ ErrorHandling::ExceptionCastProblem();\
		} \
		return boost::static_pointer_cast< RETURN_TYPE::ThisClass >( input ); \
	}

#define CONFIGURABLE_PREPARE_CAST_CONST_SMART_POINTER_MACRO( METHOD_NAME, RETURN_TYPE, INPUT_TYPE )	\
	static RETURN_TYPE::ConstPtr \
	METHOD_NAME( INPUT_TYPE::ConstPtr input ) \
	{	if( dynamic_cast< RETURN_TYPE::ConstSimplePointer >( input.get() ) == NULL ) { \
			_THROW_ ErrorHandling::ExceptionCastProblem();\
		} \
		return boost::static_pointer_cast< RETURN_TYPE::ConstThisClass >( input ); \
	}

#define CONFIGURABLE_PREPARE_CAST_METHODS_MACRO( METHOD_NAME, RETURN_TYPE, INPUT_TYPE )	\
	CONFIGURABLE_PREPARE_CAST_REFERENCE_MACRO( METHOD_NAME, RETURN_TYPE, INPUT_TYPE ) \
	CONFIGURABLE_PREPARE_CAST_CONST_REFERENCE_MACRO( METHOD_NAME, RETURN_TYPE, INPUT_TYPE ) \
	CONFIGURABLE_PREPARE_CAST_SMART_POINTER_MACRO( METHOD_NAME, RETURN_TYPE, INPUT_TYPE ) \
	CONFIGURABLE_PREPARE_CAST_CONST_SMART_POINTER_MACRO( METHOD_NAME, RETURN_TYPE, INPUT_TYPE )


}/*namespace Imaging*/
}/*namespace M4D*/

/** @} */

#endif /*_DATA_SET_DEFINITION_TOOLS_H*/


/** @} */


