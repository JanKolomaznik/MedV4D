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
	uint16 m_bitsAllocated;
	uint16 m_bitsStored;
	uint16 m_highBit;

	void *m_pixelData;
	
	//friend class M4DMoveService;
	
	Status m_status;

public:	
	void *m_dataset;

	void Init( void) throw (...);

	inline bool IsLoaded( void) { return m_status == Loaded; }

	// load & save
	void Load( const string &path) throw (...);
	void Save( const string &path)	throw (...);

	////////////////////////////////////////////////////////////
	// simple methods that returns values of each tag
	// patient level
	string GetPatientName( void) throw (...);
	bool	GetPatientSex( void) throw (...);
	string GetPatientBirthDate( void) throw (...);
	string GetPatientId( void) throw (...);

	// study level
	string GetStudyInstanceUID( void) throw (...);

	// series level
	string GetSeriesInstanceUID ( void) throw (...);
	string GetModality( void) throw (...);

	// image level
	void * GetPixelData( void) throw (...);
	// ...
	
	////////////////////////////////////////////////////////////

	DicomObj();
};

#endif