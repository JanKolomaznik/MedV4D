
#include "dcmtk/dcmdata/dcdeftag.h"
#include "dcmtk/dcmdata/dcfilefo.h"

#include "Common.h"
#include "M4DDICOMServiceProvider.h"

#include "FromStreamConverter.h"

using namespace M4D::Dicom;
using namespace M4D::DicomInternal;

///////////////////////////////////////////////////////////////////////

DcmProvider::DicomObj::DicomObj()
{
	m_dataset = NULL;
	m_status = Loading;
}

///////////////////////////////////////////////////////////////////////

void
DcmProvider::DicomObj::Save( const string &path)	
	
{
	if( m_dataset == NULL)
		return;

	//DcmFileFormat file;
	DcmDataset *dataSet = static_cast<DcmDataset *>(m_dataset);

	//E_EncodingType    opt_sequenceType = EET_ExplicitLength;
	//E_PaddingEncoding opt_paddingType = EPD_withoutPadding;
	//E_GrpLenEncoding  opt_groupLength = EGL_recalcGL;
	//unsigned int opt_itempad = 0;
	//unsigned int opt_filepad = 0;
	//OFBool            opt_useMetaheader = OFTrue;

	OFCondition cond = dataSet->saveFile( path.c_str());
	if (cond.bad())
	{
		LOG( "Cannot write file:" << path);
		throw ExceptionBase();
	}
}

///////////////////////////////////////////////////////////////////////

void
DcmProvider::DicomObj::Load( const string &path) 
{
	DcmDataset *dataSet = new DcmDataset();
	OFCondition cond = dataSet->loadFile( path.c_str());

	if (cond.bad())
	{
		LOG( "Cannot load the file: " << path);
		throw ExceptionBase();
	}

	m_dataset = (void *)dataSet;
}

///////////////////////////////////////////////////////////////////////

void
DcmProvider::DicomObj::GetTagValue( 
	uint16 group, uint16 tagNum, string &s) 
{
	OFString str;
	static_cast<DcmDataset *>(m_dataset)->findAndGetOFString( 
		DcmTagKey( group, tagNum), str );
	s = str.c_str();
}

///////////////////////////////////////////////////////////////////////

void
DcmProvider::DicomObj::GetTagValue(
	uint16 group, uint16 tagNum, int32 &i) 
{
	static_cast<DcmDataset *>(m_dataset)->findAndGetSint32( 
		DcmTagKey( group, tagNum), (Sint32 &)i );
}

///////////////////////////////////////////////////////////////////////

void
DcmProvider::DicomObj::GetTagValue( 
	uint16 group, uint16 tagNum, float &f) 
{
	static_cast<DcmDataset *>(m_dataset)->findAndGetFloat32( 
		DcmTagKey( group, tagNum), f );
}

///////////////////////////////////////////////////////////////////////

void
DcmProvider::DicomObj::Init()
{
	DcmDataset* dataSet = static_cast<DcmDataset *>(m_dataset);
	if( dataSet == NULL)
		throw ExceptionBase("No data available!");

	// get image with & height
	dataSet->findAndGetUint16( DCM_Columns, m_width);
	dataSet->findAndGetUint16( DCM_Rows, m_height);

  {
    uint16 bitsStored;
    // get bits stored to find out what data type will be used to hold data
	  dataSet->findAndGetUint16( DCM_BitsStored, bitsStored);

    if( bitsStored <= 8)
		  m_pixelSize = 1;
	  else if( bitsStored > 8 && bitsStored <= 16)
		  m_pixelSize = 2;
	  else if( bitsStored > 16)
		  m_pixelSize = 4;
	  else
		  throw ExceptionBase( "Bad Pixel Size");
  }
	
  // get order in set
  OFString str;
  dataSet->findAndGetOFString( DCM_InstanceNumber, str);
  m_orderInSet = (uint16)strtoul( str.c_str(), NULL, 10);

	// try to get data
	const uint16 *data;

	// since we are using only implicit transfer syntax array are 16bit.
	OFCondition cond = dataSet->findAndGetUint16Array( 
			DCM_PixelData, data, NULL);
	if( cond.bad() )
	{
		D_PRINT( "Cannot obtain pixel data!");
		m_status = Failed;
	}
	else
		m_status = Loaded;
}

///////////////////////////////////////////////////////////////////////

template <typename T>
void
DcmProvider::DicomObj::FlushIntoArray( const T *dest) 
{
	DcmDataset* dataSet = static_cast<DcmDataset *>(m_dataset);

	if( dataSet == NULL)
		throw ExceptionBase("No data available!");

	uint16 bitsAllocated, highBit, pixelRepresentation, bitsStored;
	// get other needed pixel attribs
  dataSet->findAndGetUint16( DCM_BitsStored, bitsStored);
	dataSet->findAndGetUint16( DCM_BitsAllocated, bitsAllocated);
	dataSet->findAndGetUint16( DCM_HighBit, highBit);
	dataSet->findAndGetUint16( DCM_PixelRepresentation, pixelRepresentation);

  if( pixelRepresentation > 0)  m_signed = true;
  else m_signed = false;

	const uint16 *data;

	// since we are using only implicit transfer syntax array are 16bit.
	OFCondition cond = dataSet->findAndGetUint16Array( 
			DCM_PixelData, data, NULL);
	if( cond.bad() )
	{
		throw ExceptionBase( "Cannot obtain pixel data!");
	}

  if( bitsAllocated == 8 &&
		highBit == 7 &&
		bitsStored == 8)	
  {
    if( sizeof( T) != 2)
      throw ExceptionBase( "Destination type is not corresponding with PixelSize!");
  }

	else if( //pixelRepresentation == 0 &&
		bitsAllocated == 16 &&
		highBit == 15 &&
		bitsStored == 16)	
		// basic setting. TODO: rewrite to be able to accept others settings
	{
    // check if was passed right type
    if( sizeof( T) != 2)
      throw ExceptionBase( "Destination type is not corresponding with PixelSize!");

		register uint16 i, j;
		register uint16 *destIter = (uint16 *)dest;
		register uint16 *srcIter = (uint16 *)data;
		// copy that
		for( i=0; i<GetHeight(); i++)
			for( j=0; j<GetWidth(); j++)
			{
				*destIter = *srcIter;
				destIter++;	srcIter++;
			}
	}

  else if( bitsAllocated == 32 &&
		highBit == 31 &&
		bitsStored == 32)	
  {
    if( sizeof( T) != 4)
      throw ExceptionBase( 
        "Destination type is not corresponding with PixelSize!");
  }

  // none of above. Custom DICOM stream.
  else
  {
    FromStreamConverter<T> conv(
      bitsAllocated, highBit, bitsStored, (uint16 *)data);

    register uint16 i, j;
		register T *destIter = (T *)dest;
		// copy that
		for( i=0; i<GetHeight(); i++)
			for( j=0; j<GetWidth(); j++)
			{
        *destIter = conv.GetItem();
				destIter++;
			}
  }
}

///////////////////////////////////////////////////////////////////////

//void
//DcmProvider::DicomObj::EncodePixelValue16Aligned( 
//	uint16 , uint16 , uint16 &)
//{
//	//// pixelCell if the image are stored in rows. 1st row, then 2nd ...
//	//uint8 *fingerToStream = (uint8 *) ( m_pixelData) + ( 	// base
//	//	((y-1) * m_width + (x-1)) *		// order of a cell in stream
//	//	(m_bitsAllocated << 3)			// size of a cell in bytes
//	//	);
//
//	//// little endian suposed
//	//val = (*fingerToStream + (*(fingerToStream + 1) << 8)) >>
//	//	(m_highBit+1-m_bitsStored);	// align to begin in pixelCell
//}