#ifndef M4D_DICOM_OBJECT
#define M4D_DICOM_OBJECT

class M4DDcmProvider::DicomObj
{
private:

	typedef enum eStatus {
		Loaded,
		Loading,
		Failed,
	} Status;

	// image info
	uint16 m_width, m_height;
	uint16 m_bitsStored;
	
	Status m_status;

	void EncodePixelValue16Aligned( uint16 x, uint16 y, uint16 &val);

public:

	void *m_dataset;

	// defines size of pixel in bits
	typedef enum ePxlSz {
		bit8,
		bit16,
		bit32,
	} PixelSize;

	// image information members
	PixelSize GetPixelSize( void);
	inline uint16 GetWidth( void) { return m_width; }
	inline uint16 GetHeight( void) { return m_height; }

	/**
	 *	Should prepare data to be easily handled. Convert from special dicom
	 *	data stream to steam of normal data types like uint16. See DICOM spec.
	 *	08-05pu.pdf (AnnexD).
	 */
	void FlushIntoArray( const uint16 *dest) throw (...);

	inline bool IsLoaded( void) { return m_status == Loaded; }

	// load & save
	void Load( const string &path) throw (...);
	void Save( const string &path)	throw (...);

	/**
	 *	Called when image arrive
	 */
	void Init();

	// methods to get value from data set container
	void GetTagValue( uint16 group, uint16 tagNum, string &) throw (...);
	void GetTagValue( uint16 group, uint16 tagNum, int32 &) throw (...);
	void GetTagValue( uint16 group, uint16 tagNum, float &) throw (...);	

	// image level
	/**
	 *	Returns pointer to array that pixels of the image are stored.
	 *	Array should be casted to the right type according value returned
	 *	by GetPixelSize() method.
	 */
	void* GetPixelData( void) throw (...);
	// ...
	
	////////////////////////////////////////////////////////////

	DicomObj();
};

#endif