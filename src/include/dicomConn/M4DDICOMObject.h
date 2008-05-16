#ifndef M4D_DICOM_OBJECT
#define M4D_DICOM_OBJECT

#include <string>

namespace M4D
{
namespace Dicom {

  /**
   *  VR (value representation): see DICOM doc ([ver]_05.pdf chapter 6.2)
   */
	class DcmProvider::DicomObj
	{
	private:

		enum Status {
			Loaded,
			Loading,
			Failed,
		};

		// image info
		uint16 m_width, m_height;
		uint8 m_pixelSize;
    bool m_signed;
		uint16 m_orderInSet;

		Status m_status;

	public:

		void *m_dataset;

		// image information members
    inline uint8 GetPixelSize( void) { return m_pixelSize; }
		inline uint16 GetWidth( void) { return m_width; }
		inline uint16 GetHeight( void) { return m_height; }
    inline bool IsDataSigned( void) { return m_signed; }

		/**
		 *	Should prepare data to be easily handled. Convert from special dicom
		 *	data stream to steam of normal data types like uint16. See DICOM spec.
		 *	08-05pu.pdf (AnnexD).
     *  Type T should corespond to size of pixel determined from GetPixelSize
     *  method or exception will be thrown.
		 */

    template< typename T>
		void FlushIntoArray( const T *dest);

    void FlushIntoArrayNTID( void*dest, int elementTypeID );

		inline bool IsLoaded( void) { return m_status == Loaded; }

    /**
     *  Returns order number in set according that images can be sorted
     *  in order thei were accuired by the mashine.
     *  Currently InstanceNumber tag (0020,0013) is used.
     */
		inline uint16 OrderInSet( void) { return m_orderInSet; }

		// load & save
		void Load( const std::string &path);
		void Save( const std::string &path)	;

		/**
		 *	Called when image arrive
		 */
		void Init();

		// methods to get value from data set container
		void GetTagValue( uint16 group, uint16 tagNum, std::string &) ;
		void GetTagValue( uint16 group, uint16 tagNum, int32 &) ;
		void GetTagValue( uint16 group, uint16 tagNum, float &) ;	

		// image level
		/**
		 *	Returns pointer to array that pixels of the image are stored.
		 *	Array should be casted to the right type according value returned
		 *	by GetPixelSize() method.
		 */
		void* GetPixelData( void) ;
		// ...
		
		////////////////////////////////////////////////////////////

		DicomObj();
	};

}
}
#endif

