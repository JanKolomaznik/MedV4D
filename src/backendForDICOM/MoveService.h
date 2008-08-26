#ifndef M4DDICOM_MOVE_SERVICE
#define M4DDICOM_MOVE_SERVICE

/**
 *  Implements C-MOVE service to DICOM server. Its purpose is to move data files (records) from DICOM server. 
 *  There are two main functions that retrive data files. One retrive one SINGLE image. The image is specified by unique IDs on all levels (patient, study, serie, image). The other retrieve all images from specified serie (SET). That means specification of IDs on all levels but image.
 *  Process decription in a nutshell: Client (SCU) establish assotiation to sever (SCP), send query dataSet, server find matching image files, then establish another subassotiation (as SCU) with calling client (that plays SCP role) and transmit data files over the subassotiation. For more details see DICOM doc ([ver]_08.pdf chapter 9.1.4) and coresponding annexes).
 */

using namespace M4D::Dicom;

namespace M4D
{
namespace DicomInternal 
{

class MoveService : AbstractService
{
private:

  friend class M4D::Dicom::DcmProvider;

  /**
   *  The two data retrival kinds definitions. According params of this type are the kinds of retrival distinguished in callbacks.
   */
	enum eCallType {
		SINGLE_IMAGE,		// only single image -> retype to DicomObj
		IMAGE_SET			// retype to vector<DicomObj>
	};

  /**
   *  Support structure for whole SET retrival that are passed to MoveSupport. It contains result container and onLoaded callback function pointer. Because actual instances are created later in callbacks so onLoaded function pointer should go along result data container.
   */
  struct ImageSetData 
  {
    ImageSetData(
      DcmProvider::DicomObjSet *result_,
      DicomObj::ImageLoadedCallback on_loaded_)
      : result(result_)
      , on_loaded( on_loaded_)
    {
    }

    DcmProvider::DicomObjSet *result;
    DicomObj::ImageLoadedCallback on_loaded;
  };

  /// Prepares query dataSet.
	void GetQuery( 
		DcmDataset **query,
		const string *patientID,
		const string *studyID,
		const string *setID,
		const string *imageID);

  // Supporting callbacks that cooperates with DCMTK functions ...
	void MoveSupport( DcmDataset *query, void *data,
		enum eCallType type) ;
  // ...
	static void
	MoveCallback(void *callbackData, T_DIMSE_C_MoveRQ *request,
		int responseCount, T_DIMSE_C_MoveRSP *response);
  // ...
	static void
	SubAssocCallback(void *subOpCallbackData,
        T_ASC_Network *aNet, T_ASC_Association **subAssoc);
  // ...
	inline static void
	SubAssocCallbackSupp(void *subOpCallbackData,
        T_ASC_Network *aNet, T_ASC_Association **subAssoc, eCallType type);
  // ...
	static void
	SubAssocCallbackWholeSet(void *subOpCallbackData,
        T_ASC_Network *aNet, T_ASC_Association **subAssoc);
  // ...
	static void AcceptSubAssoc(
		T_ASC_Network * aNet, T_ASC_Association ** assoc) ;
  // ...
	static void	SubTransferOperationSCP(
		T_ASC_Association **subAssoc, 
		void *dicomOBJRef, eCallType type) ;
  // ...
	static void StoreSCPCallback(    
		void *callbackData,					/* in */
		T_DIMSE_StoreProgress *progress,    /* progress state */
		T_DIMSE_C_StoreRQ *req,             /* original store request */
		char *imageFileName, 
		DcmDataset **imageDataSet, /* being received into */
		/* out */
		T_DIMSE_C_StoreRSP *rsp,            /* final store response */
		DcmDataset **statusDetail);

  // ctor & dtor
	MoveService();
	~MoveService();

  // Moves one SINGLE image from server
	void MoveImage( 
		const string &patientID,
		const string &studyID,
		const string &setID,
		const string &imageID,
		DicomObj &rs);

  // Moves the whole image serie from server
	void MoveImageSet(
		const string &patientID,
		const string &studyID,
		const string &serieID,
    DcmProvider::DicomObjSet &result,
    DicomObj::ImageLoadedCallback on_loaded);
	
};

} // namespace
}

#endif

