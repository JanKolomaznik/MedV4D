
#include <string>

using namespace std;

#include "dcmtk/dcmdata/dcdeftag.h"
#include "dcmtk/dcmdata/dcfilefo.h"

#include "main.h"
#include "M4DDICOMServiceProvider.h"

///////////////////////////////////////////////////////////////////////

M4DDcmProvider::DicomObj::DicomObj()
{
	m_dataset = NULL;
	m_status = Loading;
}

///////////////////////////////////////////////////////////////////////

void
M4DDcmProvider::DicomObj::Save( const string &path)	
	throw (...)
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
		throw new bad_exception();
	}
}

///////////////////////////////////////////////////////////////////////

void
M4DDcmProvider::DicomObj::Load( const string &path) throw (...)
{
	DcmDataset *dataSet = new DcmDataset();
	OFCondition cond = dataSet->loadFile( path.c_str());

	if (cond.bad())
	{
		LOG( "Cannot load the file: " << path);
		throw new bad_exception();
	}

	m_dataset = (void *)dataSet;
}

///////////////////////////////////////////////////////////////////////

void
M4DDcmProvider::DicomObj::GetTagValue( 
	uint16 group, uint16 tagNum, string &s) throw (...)
{
	OFString str;
	static_cast<DcmDataset *>(m_dataset)->findAndGetOFString( 
		DcmTagKey( group, tagNum), str );
	s = str.c_str();
}

///////////////////////////////////////////////////////////////////////

void
M4DDcmProvider::DicomObj::GetTagValue(
	uint16 group, uint16 tagNum, int32 &i) throw (...)
{
	static_cast<DcmDataset *>(m_dataset)->findAndGetSint32( 
		DcmTagKey( group, tagNum), (Sint32 &)i );
}

///////////////////////////////////////////////////////////////////////

void
M4DDcmProvider::DicomObj::GetTagValue( 
	uint16 group, uint16 tagNum, float &f) throw (...)
{
	static_cast<DcmDataset *>(m_dataset)->findAndGetFloat32( 
		DcmTagKey( group, tagNum), f );
}

///////////////////////////////////////////////////////////////////////

M4DDcmProvider::DicomObj::PixelSize
M4DDcmProvider::DicomObj::GetPixelSize( void)
{
	if( m_bitsStored <= 8)
		return PixelSize::bit8;
	else if( m_bitsStored > 8 && m_bitsStored <= 16)
		return PixelSize::bit16;
	else if( m_bitsStored > 16)
		return PixelSize::bit32;
}

///////////////////////////////////////////////////////////////////////

void
M4DDcmProvider::DicomObj::Init()
{
	DcmDataset* dataSet = static_cast<DcmDataset *>(m_dataset);
	if( dataSet == NULL)
		throw new bad_exception("No data available!");

	// get image with & height
	dataSet->findAndGetUint16( DCM_Columns, m_width);
	dataSet->findAndGetUint16( DCM_Rows, m_height);

	// get bits stored to find out what data type will be used to hold data
	dataSet->findAndGetUint16( DCM_BitsStored, m_bitsStored);

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
M4DDcmProvider::DicomObj::FlushIntoArray( const uint16 *dest) throw (...)
{
	DcmDataset* dataSet = static_cast<DcmDataset *>(m_dataset);

	if( dataSet == NULL)
		throw new bad_exception("No data available!");

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
		throw new bad_exception( "Cannot obtain pixel data!");
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
M4DDcmProvider::DicomObj::EncodePixelValue16Aligned( 
	uint16 x, uint16 y, uint16 &val)
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