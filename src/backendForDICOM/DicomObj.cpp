
#include "dcmtk/dcmdata/dcdeftag.h"
#include "dcmtk/dcmdata/dcfilefo.h"

#include "Common.h"
#include "M4DDICOMServiceProvider.h"

namespace M4D
{
using namespace ErrorHandling;

namespace Dicom {

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

DcmProvider::DicomObj::PixelSize
DcmProvider::DicomObj::GetPixelSize( void)
{
	if( m_bitsStored <= 8)
		return bit8;
	else if( m_bitsStored > 8 && m_bitsStored <= 16)
		return bit16;
	else if( m_bitsStored > 16)
		return bit32;
	else
		throw ExceptionBase( "Bad Pixel Size");
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

	// get bits stored to find out what data type will be used to hold data
	dataSet->findAndGetUint16( DCM_BitsStored, m_bitsStored);

  // get order in set
  dataSet->findAndGetUint16( DCM_InstanceNumber, m_orderInSet);

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

void
DcmProvider::DicomObj::FlushIntoArray( const uint16 *dest) 
{
	DcmDataset* dataSet = static_cast<DcmDataset *>(m_dataset);

	if( dataSet == NULL)
		throw ExceptionBase("No data available!");

	uint16 bitsAllocated, highBit, pixelRepresentation;
	// get other needed pixel attribs
	dataSet->findAndGetUint16( DCM_BitsAllocated, bitsAllocated);
	dataSet->findAndGetUint16( DCM_HighBit, highBit);
	dataSet->findAndGetUint16( DCM_PixelRepresentation, pixelRepresentation);

	const uint16 *data;

	// since we are using only implicit transfer syntax array are 16bit.
	OFCondition cond = dataSet->findAndGetUint16Array( 
			DCM_PixelData, data, NULL);
	if( cond.bad() )
	{
		throw ExceptionBase( "Cannot obtain pixel data!");
	}

	if( pixelRepresentation == 0 &&
		bitsAllocated == 16 &&
		highBit == 15 &&
		m_bitsStored == 16)	
		// basic setting. TODO: rewrite to be able to accept others settings
	{
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
}	

///////////////////////////////////////////////////////////////////////

void
DcmProvider::DicomObj::EncodePixelValue16Aligned( 
	uint16 , uint16 , uint16 &)
{
	//// pixelCell if the image are stored in rows. 1st row, then 2nd ...
	//uint8 *fingerToStream = (uint8 *) ( m_pixelData) + ( 	// base
	//	((y-1) * m_width + (x-1)) *		// order of a cell in stream
	//	(m_bitsAllocated << 3)			// size of a cell in bytes
	//	);

	//// little endian suposed
	//val = (*fingerToStream + (*(fingerToStream + 1) << 8)) >>
	//	(m_highBit+1-m_bitsStored);	// align to begin in pixelCell
}

} // namespace
}