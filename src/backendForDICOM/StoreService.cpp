#include "dcmtk/dcmnet/dimse.h"
#include "dcmtk/dcmnet/diutil.h"
#include "dcmtk/dcmdata/dcdeftag.h"
#include "dcmtk/dcmdata/dcdict.h"		// data dictionary

#include "main.h"

#include "M4DDicomAssoc.h"
#include "M4DDICOMServiceProvider.h"
#include "AbstractService.h"
#include "StoreService.h"

#ifdef ON_THE_FLY_COMPRESSION
#include "dcmtk/dcmjpeg/djdecode.h"  /* for dcmjpeg decoders */
#include "dcmtk/dcmjpeg/djencode.h"  /* for dcmjpeg encoders */
#include "dcmtk/dcmdata/dcrledrg.h"  /* for DcmRLEDecoderRegistration */
#include "dcmtk/dcmdata/dcrleerg.h"  /* for DcmRLEEncoderRegistration */
#endif

namespace M4D
{
using namespace ErrorHandling;
using namespace Dicom;

namespace DicomInternal {

///////////////////////////////////////////////////////////////////////

uint16 StoreService::seriesCounter = 0;
uint16 StoreService::imageCounter = 0;
OFString StoreService::seriesInstanceUID;
OFString StoreService::seriesNumber;
OFString StoreService::accessionNumber;

///////////////////////////////////////////////////////////////////////

StoreService::StoreService()
{
	/* make sure data dictionary is loaded */
    if (!dcmDataDict.isDictionaryLoaded()) {
        D_PRINT(
			"Warning: no data dictionary loaded, check environment variable:"
            << DCM_DICT_ENVIRONMENT_VARIABLE);
		throw ExceptionBase();
    }

	/* initialize network, i.e. create an instance of T_ASC_Network*. */
	#define TIME_OUT 30

	OFCondition cond = ASC_initializeNetwork(
		NET_REQUESTOR, 0, TIME_OUT, &m_net);
	if (cond.bad()) {
		DimseCondition::dump(cond);
		exit(1);
	}

	// create assotiation
	m_assocToServer = new M4DDicomAssociation( "store");

#ifdef ON_THE_FLY_COMPRESSION
    // register global JPEG decompression codecs
    DJDecoderRegistration::registerCodecs();

    // register global JPEG compression codecs
    DJEncoderRegistration::registerCodecs();

    // register RLE compression codec
    DcmRLEEncoderRegistration::registerCodecs();

    // register RLE decompression codec
    DcmRLEDecoderRegistration::registerCodecs();
#endif
}

///////////////////////////////////////////////////////////////////////

StoreService::~StoreService()
{
#ifdef ON_THE_FLY_COMPRESSION
    // deregister JPEG codecs
    DJDecoderRegistration::cleanup();
    DJEncoderRegistration::cleanup();

    // deregister RLE codecs
    DcmRLEDecoderRegistration::cleanup();
    DcmRLEEncoderRegistration::cleanup();
#endif

	delete m_assocToServer;
}

///////////////////////////////////////////////////////////////////////

void
StoreService::StoreSCP( DcmDataset *data) 
{
	DIC_US msgId = m_assocToServer->GetAssociation()->nextMsgID++;
    T_ASC_PresentationContextID presId;
    T_DIMSE_C_StoreRQ req;
    T_DIMSE_C_StoreRSP rsp;
    DIC_UI sopClass;
    DIC_UI sopInstance;
    DcmDataset *statusDetail = NULL;

    /* figure out which SOP class and SOP instance is encapsulated in the file */
    if (!DU_findSOPClassAndInstanceInDataSet(data,
        sopClass, sopInstance)) 
	{
        LOG("No SOP Class & Instance UIDs in data set");
        throw ExceptionBase();
    }

    /* figure out which of the accepted presentation contexts should be used */
    DcmXfer filexfer(data->getOriginalXfer());

    /* special case: if the file uses an unencapsulated transfer syntax (uncompressed
     * or deflated explicit VR) and we prefer deflated explicit VR, then try
     * to find a presentation context for deflated explicit VR first.
     */
    if (filexfer.isNotEncapsulated() &&
		m_assocToServer->GetAssocAddress()->transferSyntax ==
		EXS_DeflatedLittleEndianExplicit)
    {
        filexfer = EXS_DeflatedLittleEndianExplicit;
    }

    if (filexfer.getXfer() != EXS_Unknown) 
		presId = ASC_findAcceptedPresentationContextID(
			m_assocToServer->GetAssociation(), sopClass, filexfer.getXferID());
    else 
		presId = ASC_findAcceptedPresentationContextID(
			m_assocToServer->GetAssociation(), sopClass);

    if (presId == 0) {
        const char *modalityName = dcmSOPClassUIDToModality(sopClass);
        if (!modalityName) modalityName = dcmFindNameOfUID(sopClass);
        if (!modalityName) modalityName = "unknown SOP class";
		{
			D_PRINT("No presentation context for: ("
				<< modalityName << ")" << sopClass);
			throw ExceptionBase();
		}
    }

    /* dump general information concerning transfer syntaxes */
    DcmXfer fileTransfer( data->getOriginalXfer());
    T_ASC_PresentationContext pc;
    //ASC_findAcceptedPresentationContext(m_assocToServer->, presId, &pc);
    DcmXfer netTransfer(pc.acceptedTransferSyntax);

    LOG("Transfer: " << dcmFindNameOfUID(fileTransfer.getXferID())
		<< " -> " << dcmFindNameOfUID(netTransfer.getXferID()) );    

    /* prepare the transmission of data */
    bzero((char*)&req, sizeof(req));
    req.MessageID = msgId;
    strcpy(req.AffectedSOPClassUID, sopClass);
    strcpy(req.AffectedSOPInstanceUID, sopInstance);
    req.DataSetType = DIMSE_DATASET_PRESENT;
    req.Priority = DIMSE_PRIORITY_LOW;

    LOG("Store SCU RQ: MsgID " << msgId <<
		", (" << dcmSOPClassUIDToModality(sopClass) << ")");

#define TIMEOUT 30
    /* finally conduct transmission of data */
	OFCondition cond = DIMSE_storeUser(
		m_assocToServer->GetAssociation(), presId, &req,
        NULL, data, ProgressCallback, NULL,
        DIMSE_NONBLOCKING, TIMEOUT,
		&rsp, &statusDetail, NULL, data->getLength());

    /*
     * If store command completed normally, with a status
     * of success or some warning then the image was accepted.
     */
    if ( ! (cond == EC_Normal && (rsp.DimseStatus == STATUS_Success || 
		DICOM_WARNING_STATUS(rsp.DimseStatus))) )
	{
        LOG( "Image was not accepted!");
		throw ExceptionBase();
    }

	if( cond != EC_Normal)
	{
		LOG( "C_STORE operation failed!");
		throw ExceptionBase();
	}

    /* dump status detail information if there is some */
    if (statusDetail != NULL) {
        D_PRINT("Status Detail:");
        statusDetail->print(LOUT);
        delete statusDetail;
    }
}

///////////////////////////////////////////////////////////////////////

OFBool
StoreService::IsaListMember(OFList<OFString>& lst, OFString& s)
{
    OFListIterator(OFString) cur = lst.begin();
    OFListIterator(OFString) end = lst.end();

    OFBool found = OFFalse;

    while (cur != end && !found) {

        found = (s == *cur);

        ++cur;
    }

    return found;
}

///////////////////////////////////////////////////////////////////////

void
StoreService::AddPresentationContext(T_ASC_Parameters *params,
    T_ASC_PresentationContextID presentationContextId,
	const OFString& abstractSyntax,
    const OFString& transferSyntax,
    T_ASC_SC_ROLE proposedRole = ASC_SC_ROLE_DEFAULT) 
{
    const char* c_p = transferSyntax.c_str();
    OFCondition cond = ASC_addPresentationContext(params, presentationContextId,
        abstractSyntax.c_str(), &c_p, 1, proposedRole);
	if( cond.bad() )
	{
		D_PRINT( "Cannot add presentatoin ctx");
		throw ExceptionBase();
	}
}

///////////////////////////////////////////////////////////////////////

void
StoreService::AddPresentationContext(T_ASC_Parameters *params,
    T_ASC_PresentationContextID presentationContextId,
	const OFString& abstractSyntax,
    const OFList<OFString>& transferSyntaxList,
    T_ASC_SC_ROLE proposedRole = ASC_SC_ROLE_DEFAULT) 
{
    // create an array of supported/possible transfer syntaxes
    const char** transferSyntaxes = new const char*[transferSyntaxList.size()];
    int transferSyntaxCount = 0;
    OFListConstIterator(OFString) s_cur = transferSyntaxList.begin();
    OFListConstIterator(OFString) s_end = transferSyntaxList.end();
    while (s_cur != s_end) {
        transferSyntaxes[transferSyntaxCount++] = (*s_cur).c_str();
        ++s_cur;
    }

    OFCondition cond = ASC_addPresentationContext(params, presentationContextId,
        abstractSyntax.c_str(), transferSyntaxes, transferSyntaxCount, proposedRole);

    delete[] transferSyntaxes;

	if( cond.bad() )
	{
		D_PRINT( "Cannot add presentation ctxs");
		throw ExceptionBase();
	}
}

///////////////////////////////////////////////////////////////////////

void
StoreService::AddStoragePresentationContexts(
	T_ASC_Parameters *params, OFList<OFString>& sopClasses) 
{
    /*
     * Each SOP Class will be proposed in two presentation contexts (unless
     * the opt_combineProposedTransferSyntaxes global variable is true).
     * The command line specified a preferred transfer syntax to use.
     * This prefered transfer syntax will be proposed in one
     * presentation context and a set of alternative (fallback) transfer
     * syntaxes will be proposed in a different presentation context.
     *
     * Generally, we prefer to use Explicitly encoded transfer syntaxes
     * and if running on a Little Endian machine we prefer
     * LittleEndianExplicitTransferSyntax to BigEndianTransferSyntax.
     * Some SCP implementations will just select the first transfer
     * syntax they support (this is not part of the standard) so
     * organise the proposed transfer syntaxes to take advantage
     * of such behaviour.
     */

	// transfer syntax we want to use
	E_TransferSyntax opt_networkTransferSyntax = EXS_LittleEndianImplicit;

    // Which transfer syntax was preferred on the command line
    OFString preferredTransferSyntax;
    if (opt_networkTransferSyntax == EXS_Unknown) {
        /* gLocalByteOrder is defined in dcxfer.h */
        if (gLocalByteOrder == EBO_LittleEndian) {
            /* we are on a little endian machine */
            preferredTransferSyntax = UID_LittleEndianExplicitTransferSyntax;
        } else {
            /* we are on a big endian machine */
            preferredTransferSyntax = UID_BigEndianExplicitTransferSyntax;
        }
    } else {
        DcmXfer xfer(opt_networkTransferSyntax);
        preferredTransferSyntax = xfer.getXferID();
    }

    OFListIterator(OFString) s_cur;
    OFListIterator(OFString) s_end;


    OFList<OFString> fallbackSyntaxes;
    fallbackSyntaxes.push_back(UID_LittleEndianExplicitTransferSyntax);
    fallbackSyntaxes.push_back(UID_BigEndianExplicitTransferSyntax);
    fallbackSyntaxes.push_back(UID_LittleEndianImplicitTransferSyntax);
    // Remove the preferred syntax from the fallback list
    fallbackSyntaxes.remove(preferredTransferSyntax);
    // If little endian implicit is preferred then we don't need any fallback syntaxes
    // because it is the default transfer syntax and all applications must support it.
    if (opt_networkTransferSyntax == EXS_LittleEndianImplicit) {
        fallbackSyntaxes.clear();
    }

    // created a list of transfer syntaxes combined from the preferred and fallback syntaxes
    OFList<OFString> combinedSyntaxes;
    s_cur = fallbackSyntaxes.begin();
    s_end = fallbackSyntaxes.end();
    combinedSyntaxes.push_back(preferredTransferSyntax);
    while (s_cur != s_end)
    {
        if (!IsaListMember(combinedSyntaxes, *s_cur)) combinedSyntaxes.push_back(*s_cur);
        ++s_cur;
    }

    //if (!opt_proposeOnlyRequiredPresentationContexts) {
    //    // add the (short list of) known storage sop classes to the list
    //    // the array of Storage SOP Class UIDs comes from dcuid.h
    //    for (int i=0; i<numberOfDcmShortSCUStorageSOPClassUIDs; i++) {
    //        sopClasses.push_back(dcmShortSCUStorageSOPClassUIDs[i]);
    //    }
    //}

    // thin out the sop classes to remove any duplicates.
    OFList<OFString> sops;
    s_cur = sopClasses.begin();
    s_end = sopClasses.end();
    while (s_cur != s_end) {
        if (! IsaListMember(sops, *s_cur)) {
            sops.push_back(*s_cur);
        }
        ++s_cur;
    }

	bool opt_combineProposedTransferSyntaxes = true;	// TODO

    // add a presentations context for each sop class / transfer syntax pair
    OFCondition cond = EC_Normal;
    T_ASC_PresentationContextID pid = 1; // presentation context id
    s_cur = sops.begin();
    s_end = sops.end();
    while (s_cur != s_end && cond.good()) {

        if (pid > 255) {
            D_PRINT("Too many presentation contexts");
            throw ExceptionBase();
        }

        if (opt_combineProposedTransferSyntaxes) {
            AddPresentationContext(params, pid, *s_cur, combinedSyntaxes);
            pid += 2;   /* only odd presentation context id's */
        } else {

            // sop class with preferred transfer syntax
            AddPresentationContext(params, pid, *s_cur, preferredTransferSyntax);
            pid += 2;   /* only odd presentation context id's */

            if (fallbackSyntaxes.size() > 0) {
                if (pid > 255) {
                    D_PRINT("Too many presentation contexts");
                    throw ExceptionBase();
                }

                // sop class with fallback transfer syntax
                AddPresentationContext(params, pid, *s_cur, fallbackSyntaxes);
                pid += 2;       /* only odd presentation context id's */
            }
        }
        ++s_cur;
    }
}

///////////////////////////////////////////////////////////////////////

int
StoreService::SecondsSince1970()
{
    time_t t = time(NULL);
    return (int)t;
}

///////////////////////////////////////////////////////////////////////

OFString
StoreService::IntToString(int i)
{
    char numbuf[32];
    sprintf(numbuf, "%d", i);
    return numbuf;
}

///////////////////////////////////////////////////////////////////////

OFString
StoreService::MakeUID(OFString basePrefix, int counter)
{
    OFString prefix = basePrefix + "." + IntToString(counter);
    char uidbuf[65];
    OFString uid = dcmGenerateUniqueIdentifier(uidbuf, prefix.c_str());
    return uid;
}

///////////////////////////////////////////////////////////////////////

OFBool
StoreService::UpdateStringAttributeValue(
	DcmItem* dataset, const DcmTagKey& key, OFString& value)
{
    DcmStack stack;
    DcmTag tag(key);

    OFCondition cond = EC_Normal;
    cond = dataset->search(key, stack, ESM_fromHere, OFFalse);
    if (cond != EC_Normal) {
        CERR << "error: updateStringAttributeValue: cannot find: " << tag.getTagName()
             << " " << key << ": "
             << cond.text() << endl;
        return OFFalse;
    }

    DcmElement* elem = (DcmElement*) stack.top();

    DcmVR vr(elem->ident());
    if (elem->getLength() > vr.getMaxValueLength()) {
        CERR << "error: updateStringAttributeValue: INTERNAL ERROR: " << tag.getTagName()
             << " " << key << ": value too large (max "
            << vr.getMaxValueLength() << ") for " << vr.getVRName() << " value: " << value << endl;
        return OFFalse;
    }

    cond = elem->putOFStringArray(value);
    if (cond != EC_Normal) {
        CERR << "error: updateStringAttributeValue: cannot put string in attribute: " << tag.getTagName()
             << " " << key << ": "
             << cond.text() << endl;
        return OFFalse;
    }

    return OFTrue;
}

///////////////////////////////////////////////////////////////////////

void
StoreService::ReplaceSOPInstanceInformation(DcmDataset* dataset)
{
#define STUDY_ID_PREFIX "SID_"		// StudyID is SH (maximum 16 chars)	
	OFString studyIDPrefix(STUDY_ID_PREFIX);

#define ACCESSION_PREFIX ""			// AccessionNumber is SH (maximum 16 chars)
	OFString accessionNumberPrefix(STUDY_ID_PREFIX);

    if (seriesInstanceUID.length() == 0) 
		seriesInstanceUID=MakeUID(SITE_SERIES_UID_ROOT, (int)seriesCounter);

    if (seriesNumber.length() == 0) 
		seriesNumber = IntToString((int)seriesCounter);
   
	if (accessionNumber.length() == 0) 
		accessionNumber = 
			accessionNumberPrefix +
			IntToString( SecondsSince1970())/* + 
			IntToString((int)studyCounter)*/;

    OFString sopInstanceUID = MakeUID(SITE_INSTANCE_UID_ROOT, (int)imageCounter);
    OFString imageNumber = IntToString((int)imageCounter);

	// print invented IDs
	D_PRINT( "Inventing Identifying Information:");
	D_PRINT( "SeriesInstanceUID=" << seriesInstanceUID << endl
        << "  SeriesNumber=" << seriesNumber << endl
        << "  SOPInstanceUID=" << sopInstanceUID << endl
        << "  ImageNumber=" << imageNumber << endl );

    UpdateStringAttributeValue(dataset, DCM_SeriesInstanceUID, seriesInstanceUID);
    UpdateStringAttributeValue(dataset, DCM_SeriesNumber, seriesNumber);
    UpdateStringAttributeValue(dataset, DCM_SOPInstanceUID, sopInstanceUID);
    UpdateStringAttributeValue(dataset, DCM_InstanceNumber, imageNumber);

    imageCounter++;
}

///////////////////////////////////////////////////////////////////////

void
StoreService::ProgressCallback(void * /*callbackData*/,
    T_DIMSE_StoreProgress *progress,
    T_DIMSE_C_StoreRQ * /*req*/)
{
    switch (progress->state) 
	{
    case DIMSE_StoreBegin:
		LOG("C-STORE begins");
        break;
    case DIMSE_StoreEnd:
		LOG("C-STORE has end");
		break;
    }
}

///////////////////////////////////////////////////////////////////////

void
StoreService::StoreObject( 
		const DcmProvider::DicomObj &objectToCopyAttribsFrom,
		DcmProvider::DicomObj &objectToStore) 
{
	DcmDataset *dataSetToStore = (DcmDataset *) objectToStore.m_dataset;
	DcmDataset *dataSetPattern = 
		(DcmDataset *) objectToCopyAttribsFrom.m_dataset;

	// copy all neccessary attribs from pattern to image
	CopyNeccessaryAttribs( dataSetPattern, dataSetToStore);

	// generate setUID & image UID
    ReplaceSOPInstanceInformation( dataSetToStore);

	StoreSCP( dataSetToStore);
}

///////////////////////////////////////////////////////////////////////

void
StoreService::CopyNeccessaryAttribs( 
					DcmDataset *source, DcmDataset *dest)
{
	char tmp[256];
	const char *tmpPtr = tmp;

	//copy patient name
	source->findAndGetString( DCM_PatientsName, tmpPtr);
	dest->putAndInsertString( DCM_PatientsName, tmp);

	//copy patient ID
	source->findAndGetString( DCM_PatientID, tmpPtr);
	dest->putAndInsertString( DCM_PatientID, tmp);

	//copy study ID
	source->findAndGetString( DCM_StudyInstanceUID, tmpPtr);
	dest->putAndInsertString( DCM_StudyInstanceUID, tmp);
}

} // namespace
}