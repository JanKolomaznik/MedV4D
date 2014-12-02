/**
 * @ingroup imaging
 * @author Jan Kolomaznik
 * @file DatasetSerializationTools.h
 * @{
 **/

#ifndef _DATASET_SERIALIZATION_TOOLS_H
#define _DATASET_SERIALIZATION_TOOLS_H


#include <memory>
#include "MedV4D/Common/Common.h"
#include "MedV4D/Imaging/DatasetClassEnum.h"
#include "MedV4D/Common/FStreams.h"
#include "MedV4D/Common/ExceptionBase.h"
/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

namespace M4D
{
namespace Imaging {

enum {
	DUMP_START_MAGIC_NUMBER		= 0xFEEDDEAF,
	DUMP_HEADER_END_MAGIC_NUMBER 	= 0xDEADBEAF,
	DUMP_END_MAGIC_NUMBER 		= 0x0DEADBEE,
	DUMP_SLICE_BEGIN_MAGIC_NUMBER	= 0xAAAAFFFF,
	DUMP_SLICE_END_MAGIC_NUMBER	= 0xBBBBFFFF,
	ACTUAL_FORMAT_VERSION 		= 1
};

class EWrongStreamBeginning : public M4D::ErrorHandling::io_error {};
class EWrongStreamEnd: public M4D::ErrorHandling::io_error {};
class EWrongFormatVersion: public M4D::ErrorHandling::io_error {};
class EWrongHeader: public M4D::ErrorHandling::io_error {};
class EWrongDatasetTypeIdentification: public M4D::ErrorHandling::io_error {};
class EWrongDatasetType: public M4D::ErrorHandling::io_error {};


inline void
SerializeHeader ( M4D::IO::OutStream &stream, DatasetType datasetType )
{
	stream.Put<uint32> ( DUMP_START_MAGIC_NUMBER );
	stream.Put<uint32> ( ACTUAL_FORMAT_VERSION );

	stream.Put<uint32> ( datasetType );
}

inline uint32
DeserializeHeader ( M4D::IO::InStream &stream )
{
	uint32 startMAGIC = 0;
	uint32 datasetType = 0;
	uint32 formatVersion = 0;

	//Read stream header
	stream.Get<uint32> ( startMAGIC );
	if ( startMAGIC != DUMP_START_MAGIC_NUMBER ) {
		_THROW_ EWrongStreamBeginning();
	}

	stream.Get<uint32> ( formatVersion );
	if ( formatVersion != ACTUAL_FORMAT_VERSION ) {
		_THROW_ EWrongFormatVersion();
	}

	stream.Get<uint32> ( datasetType );

	return datasetType;
}




}/*namespace Imaging*/
}/*namespace M4D*/

/** @} */

#endif /*_DATASET_SERIALIZATION_TOOLS_H*/


/** @} */
