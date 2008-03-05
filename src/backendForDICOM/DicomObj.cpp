
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

string
M4DDcmProvider::DicomObj::GetPatientName( void) throw (...)
{
	OFString s;
	static_cast<DcmDataset *>(m_dataset)->findAndGetOFString( 
		DCM_PatientsName, s );

	return string(s.c_str());
}

///////////////////////////////////////////////////////////////////////

void
M4DDcmProvider::DicomObj::Init( void) throw (...)
{
	m_status = Loaded;

	DcmDataset* dataSet = static_cast<DcmDataset *>(m_dataset);

	if( dataSet == NULL)
		throw new bad_exception("No data available!");

	// get image with & height
	dataSet->findAndGetUint16( DCM_Columns, m_width);
	dataSet->findAndGetUint16( DCM_Rows, m_height);

	// get pixel attribs
	dataSet->findAndGetUint16( DCM_BitsStored, m_bitsStored);
	dataSet->findAndGetUint16( DCM_BitsAllocated, m_bitsAllocated);
	dataSet->findAndGetUint16( DCM_HighBit, m_highBit);

	uint8 data[1024];
	const uint8 *dataPtr = data;
	unsigned long count = 10;

	OFCondition cond = dataSet->findAndGetUint8Array( DCM_PixelData, dataPtr, &count);
	if( cond.good() )
	{
		D_PRINT("DATA:::::::::::::::::");
		for( int i = 0; i < count; i++)
			if( *(dataPtr + i) != 0 && *(dataPtr + i) != 1)
				D_PRINT( ((int)*(dataPtr + i)) << "|");
		int j = 10;
	}
	else
	{
		int i = 11;
	}
}

///////////////////////////////////////////////////////////////////////

//void
//EncodePixelValue( void *ptrToPixelData)
//{
//	// get (m_bitsAllocated / 8) bytes from buff
//	uint8 *val = (uint8 *) ptrToPixelData;
//
//	int firstBit = m_highBit - m_bitsStored;
//	
//}