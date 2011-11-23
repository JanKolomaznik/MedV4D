#ifndef M4D_DICOM_OBJECT
#define M4D_DICOM_OBJECT

/**
 *  @ingroup dicom
 *  @file DcmObject.h
 *  @author Vaclav Klecanda
 */
#include "common/Common.h"
#include <vector>
#include <boost/shared_ptr.hpp>


class DcmDataset;
class DcmFileFormat;

namespace M4D
{
namespace Dicom 
{

/**
 *  @addtogroup dicom
 *  @{
 */

/// This represents a DICOM file.
/** @class M4D::Dicom::DicomObj
 *  DICOM file is structured file. Elements
 *  of that structure are pair of <tag id, tag value> ( for tags definitions see 
 *  DICOM doc ([ver]_05.pdf chapter 6.2) ). Each tag has defined also it value 
 *  representation (VR : see DICOM doc ([ver]_05.pdf chapter 6.2)) that is actualy 
 *  data type of an tag. Whole set of an pairs (data elements) is called dataSet. 
 *  In each data set is normally (but can be more) data element that stores actual data 
 *  of the file. That element's data has a special format and must be specialy encoded 
 *  (see DICOM doc ([ver]_05.pdf section 8). That special format is called DICOM stream.
 *  DicomObj can be retrieved from DICOM server,
 *  or loaded from local filesystem (disc, networkplace). Both ways through DcmProvider methods. When it is being 
 *  retrieved, Init function is called. This cause some basic information
 *  to be loaded from dataSet and OnLoaded callback is called.
 */
class DicomObj
{

public:

	/// Typedef for callbacks for events
	typedef void (*ImageLoadedCallback)(void);

	inline void 
	SetLoadedCallback( ImageLoadedCallback c) 
		{ m_loadedCallBack = c; }

	/** 
	 * Pointer to actual dataset container.  
	 **/
	mutable DcmDataset *m_dataset;

	/// support for loading & saving
	DcmFileFormat *m_fileFormat;

	//  Basic image information members.
	
	/**
	 * \return size (in bytes) of one element
	 **/
	inline uint8 
	GetPixelSize( void) { return m_pixelSize; }
	
	inline uint16 
	GetWidth( void) { return m_width; }
	
	inline uint16 
	GetHeight( void) { return m_height; }
	
	inline bool 
	IsDataSigned( void) { return m_signed; }

	void 
	GetSliceThickness( float32 &f) const;
	
	void 
	GetPixelSpacing( float32 &horizSpacing, float32 &vertSpacing) const;
	
	void 
	GetSliceLocation( float32 &location) const;
	
	void 
	GetImagePosition( float32 &x, float32 &y, float32 &z ) const;

	/// Converts from special dicom data stream
	/** to steam of normal data types like uint16. See DICOM spec.
	*	08-05pu.pdf (AnnexD).
	*  \note Type T should corespond to size of pixel determined from GetPixelSize
	*  method or NOTE: UNSPECIFIED behaviour will follow !!!!! .
	*  \param dest = destination buffer where to unpack the DICOM data stream.
	*/
	template< typename T>
	void 
	FlushIntoArray( const T *dest);

	void 
	FlushIntoArrayNTID( void*dest, int elementTypeID );

	/// Returns order number in set.
	/** according that images can be sorted
	*  in order they were accuired by the mashine.
	*  Currently SliceLocation tag is used.
	*/
	inline float32 
	OrderInSet( void) const { return m_orderInSet; }

	/**
	*  For sorting issues
	*  check even AcquisitionTime to right sort multi-scan data set
	*/
	inline bool 
	operator<( const DicomObj &b) const
		{
			std::string aTime;
			std::string bTime;

			this->GetAcquisitionTime( aTime);
			b.GetAcquisitionTime( bTime);

			// convert times to integer to be comparable ..

			// compare the times
			// if they are the same dicide according order in set
			if (aTime == bTime) {
				return OrderInSet() < b.OrderInSet();
			} else {
				// else return comparison of acquisition times to sort good
				// subsets within multi-scan
				return aTime < bTime;
			}
		}

	// load & save
	void 
	Load( const std::string &path);
	
	void 
	Save( const std::string &path);

	/// Called when image arrive. 
	/** Inits basic info to member variables.
	*  These are returned via basic image information members.
	*/
	void 
	Init();

	/// Gets values from data set container
	/**   (other that the basic) 
	*/
	void 
	GetTagValue( uint16 group, uint16 tagNum, std::string &) ;
	
	void 
	GetTagValue( uint16 group, uint16 tagNum, int32 &);
	
	void 
	GetTagValue( uint16 group, uint16 tagNum, float32 &);
	
	void 
	GetAcquisitionTime(std::string &acqTime)const;

	////////////////////////////////////////////////////////////

	DicomObj();

private:

	enum Status {
	  Loaded,
	  Loading,
	  Failed,
	};

	// image info (basic set of information retrieved from data set.
	// others are to be retrieved via GetTagValue method
	uint16		m_width;
	uint16		m_height;
	uint8		m_pixelSize;
	bool		m_signed;
	float32		m_orderInSet;

	Status		m_status;

	// on loaded event callback
	ImageLoadedCallback m_loadedCallBack;
};

/// Container for one serie of images
typedef std::vector<DicomObj> DicomObjSet;

/// shared pointer to DicomObjSet type
typedef boost::shared_ptr< DicomObjSet> DicomObjSetPtr;

}
}

#endif

/**
 * @}
 */

