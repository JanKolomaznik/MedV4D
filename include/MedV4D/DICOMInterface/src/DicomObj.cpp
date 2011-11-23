/**
 *  @ingroup dicom
 *  @file DicomObj.cpp
 *  @author Vaclav Klecanda
 */

#include "dcmtk/config/osconfig.h"
#include "dcmtk/dcmdata/dcdeftag.h"
#include "dcmtk/dcmdata/dcfilefo.h"

#include "common/Common.h"
#include "../DcmObject.h"
#include "../dicomDataStreamDecoder.h"


using namespace M4D::Dicom;

///////////////////////////////////////////////////////////////////////

DicomObj::DicomObj()
{
	m_dataset = NULL;
	m_status = Loading;
	m_loadedCallBack = NULL;
}

///////////////////////////////////////////////////////////////////////

void
DicomObj::Save( const std::string &path)	
{
	if( m_dataset == NULL)
		return;

	//DcmFileFormat file;

	//E_EncodingType    opt_sequenceType = EET_ExplicitLength;
	//E_PaddingEncoding opt_paddingType = EPD_withoutPadding;
	//E_GrpLenEncoding  opt_groupLength = EGL_recalcGL;
	//unsigned int opt_itempad = 0;
	//unsigned int opt_filepad = 0;
	//OFBool            opt_useMetaheader = OFTrue;

	OFCondition cond = m_dataset->saveFile( path.c_str());
	if (! cond.good())
	{
		LOG( "Cannot write file:" << path);
		throw ExceptionBase();
	}
}

///////////////////////////////////////////////////////////////////////

void
DicomObj::Load( const std::string &path) 
{
	m_fileFormat = new DcmFileFormat();

	OFCondition cond = m_fileFormat->loadFile( path.c_str());
	if (! cond.good() ) {
		//LOG( "Cannot load the file: " << path);
		throw ErrorHandling::EFileProblem( "Cannot load DICOM file", boost::filesystem::path( path ) );
	}

	m_dataset = m_fileFormat->getDataset();
}

///////////////////////////////////////////////////////////////////////

void
DicomObj::GetTagValue( 
	uint16 group, uint16 tagNum, std::string &s) 
{
	OFString str;
	static_cast<DcmDataset *>(m_dataset)->findAndGetOFString( 
		DcmTagKey( group, tagNum), str );
	s = str.c_str();
}

///////////////////////////////////////////////////////////////////////

void
DicomObj::GetTagValue(
	uint16 group, uint16 tagNum, int32 &i) 
{
	static_cast<DcmDataset *>(m_dataset)->findAndGetSint32( 
		DcmTagKey( group, tagNum), (Sint32 &)i );
}

///////////////////////////////////////////////////////////////////////

void
DicomObj::GetTagValue( 
	uint16 group, uint16 tagNum, float32 &f) 
{
	static_cast<DcmDataset *>(m_dataset)->findAndGetFloat32( 
		DcmTagKey( group, tagNum), f );
}

///////////////////////////////////////////////////////////////////////

void
DicomObj::GetAcquisitionTime(std::string &acqTime)const
{
  // DICOM tag Acquisition Time (0008 0032)
  OFString str;
  DcmDataset *ds = static_cast<DcmDataset *>(m_dataset);

  ds->findAndGetOFString(DcmTagKey(0x0008, 0x0032), str );
  acqTime = str.c_str();
}

///////////////////////////////////////////////////////////////////////

void
DicomObj::Init()
{
	if( m_dataset == NULL)
		throw ExceptionBase("No data available!");

	// here should be solved all cases of pixelRepresentation (tag: 0x0028, 0x0103)
	// it is connected in tranfer syntax etc. ...
	//TODO better checking
	OFCondition cond = m_dataset->chooseRepresentation(EXS_LittleEndianExplicit, NULL);
	if( !cond.good() ) {
		throw ExceptionBase( "chooseRepresentation() failed");
	}

	// get image with & height
	m_dataset->findAndGetUint16( DCM_Columns, m_width);
	m_dataset->findAndGetUint16( DCM_Rows, m_height);

	{
		uint16 bitsStored;
		// get bits stored to find out what data type will be used to hold data
		m_dataset->findAndGetUint16( DCM_BitsStored, bitsStored);

		if( bitsStored <= 8) {
			m_pixelSize = 1;
		} else if( bitsStored > 8 && bitsStored <= 16) {
			m_pixelSize = 2;
		} else if( bitsStored > 16) {
			m_pixelSize = 4;
		} else {
			throw ExceptionBase( "Bad Pixel Size");
		}
	}

	// get order in set
	OFString str;
	//m_dataset->findAndGetOFString( DCM_SliceLocation, str);
	m_dataset->findAndGetOFStringArray( DCM_ImagePositionPatient, str);
	int found = (int) str.find_last_of('\\');

	std::istringstream stream( str.substr(found+1).c_str());
	stream >> m_orderInSet;


	uint16 pixelRepresentation;
	m_dataset->findAndGetUint16( DCM_PixelRepresentation, pixelRepresentation);
	if( pixelRepresentation > 0) {
		m_signed = true;
	} else {
		m_signed = false;
	}

	// try to get data
	const uint16 *data;

	// since we are using only implicit transfer syntax array are 16bit.
	cond = m_dataset->findAndGetUint16Array( DCM_PixelData, data, NULL );
	if( cond.bad() ) {
		D_PRINT( "Cannot obtain pixel data!");
		m_status = Failed;
	} else {
		m_status = Loaded;
	}

	// call loaded callback if any
	if( m_loadedCallBack != NULL) {
		m_loadedCallBack();
	}
}

///////////////////////////////////////////////////////////////////////

template <typename T>
void
DicomObj::FlushIntoArray( const T *dest) 
{
	if( m_dataset == NULL)
		throw ExceptionBase("No data available!");

	uint16 bitsAllocated, highBit, bitsStored;
	// get other needed pixel attribs
	m_dataset->findAndGetUint16( DCM_BitsStored, bitsStored);
	m_dataset->findAndGetUint16( DCM_BitsAllocated, bitsAllocated);
	m_dataset->findAndGetUint16( DCM_HighBit, highBit);


	const uint16 *data;
	unsigned long itemCount;
	// since we are using only implicit transfer syntax array are 16bit.
	OFCondition cond = m_dataset->findAndGetUint16Array( DCM_PixelData, data, &itemCount );
	if( !cond.good() || data == NULL ) {
		throw ExceptionBase( "Cannot obtain pixel data!");
	}
		D_PRINT( "Data prepared for flush = " << sizeof( uint16 ) * itemCount );
	
	if( bitsAllocated == 8 && highBit == 7 && bitsStored == 8 ) {// 1 byte aligned DICOM stream
		memcpy( (void *) dest, (void *) data, GetWidth() * GetHeight() );
	} else if( bitsAllocated == 16 && highBit == 15 && bitsStored == 16 ) {// 2bytes aligned DICOM stream
		memcpy( (void *) dest, data, GetWidth() * GetHeight() * sizeof(T) );
		//register uint16 i, j;
		//register uint16 *destIter = (uint16 *)dest;
		//register uint16 *srcIter = (uint16 *)data;
		//// copy that
		//for( i=0; i<GetHeight(); i++)
		//	for( j=0; j<GetWidth(); j++)
		//	{
		//		*destIter = *srcIter;
		//		destIter++;	srcIter++;
		//	}
	} else if( bitsAllocated == 32 && highBit == 31 && bitsStored == 32) {
		throw ExceptionBase( "Not supported DICOM stream setup");
	} else { // none of above. Custom DICOM stream.
		DicomDataStreamDecoder decoder(	bitsAllocated, highBit, bitsStored, m_signed, (uint16 *)data );

		T *destIter = (T *)dest;
		// copy that
		for( size_t i=0; i<GetHeight(); ++i ) {
			for( size_t j=0; j<GetWidth(); ++j ) {
				*destIter = decoder.GetItem<T>();
				++destIter;
			}
		}
	}
}

void 
DicomObj::FlushIntoArrayNTID( void* dest, int elementTypeID )
{ 
	INTEGER_TYPE_TEMPLATE_SWITCH_MACRO( 
		elementTypeID, FlushIntoArray<TTYPE>( (const TTYPE*) dest ) );
}

///////////////////////////////////////////////////////////////////////

void
DicomObj::GetSliceThickness( float32 &f) const
{
	if( m_dataset == NULL)
		throw ExceptionBase("No data available!");

	OFString str;
	m_dataset->findAndGetOFString( DCM_SliceThickness, str);
	std::istringstream stream( str.c_str());
	stream >> f;
}

///////////////////////////////////////////////////////////////////////

void
DicomObj::GetPixelSpacing( float32 &horizSpacing, float32 &vertSpacing) const
{
	if( m_dataset == NULL)
		throw ExceptionBase("No data available!");

	OFString s;
	m_dataset->findAndGetOFStringArray( DCM_PixelSpacing, s);

	std::istringstream stream( s.c_str());
	stream >> horizSpacing;
	stream.seekg( stream.cur);
	stream >> vertSpacing;
}

///////////////////////////////////////////////////////////////////////

void
DicomObj::GetSliceLocation( float32 &location) const
{
	if( m_dataset == NULL)
		throw ExceptionBase("No data available!");

	OFString str;
	m_dataset->findAndGetOFString( DCM_SliceLocation, str);
	std::istringstream stream( str.c_str());
	stream >> location;
}

///////////////////////////////////////////////////////////////////////

void 
DicomObj::GetImagePosition( float32 &x, float32 &y, float32 &z ) const
{
	if( m_dataset == NULL)
		throw ExceptionBase("No data available!");

	OFString s;
	m_dataset->findAndGetOFStringArray( DCM_ImagePositionPatient, s);

	std::istringstream stream( s.c_str());
	stream >> x;
	stream.seekg( stream.cur);
	stream >> y;
	stream.seekg( stream.cur);
	stream >> z;
}

/**
 * @}
 */

