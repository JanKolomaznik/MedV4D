/**
 *  @ingroup dicom
 *  @file AbstractService.cpp
 *  @author Vaclav Klecanda
 */
#include "dcmtk/config/osconfig.h"
#include "dcmtk/dcmnet/dimse.h"
#include "dcmtk/dcmnet/diutil.h"
#include "dcmtk/dcmdata/dcdeftag.h"

#include "MedV4D/Common/Common.h"

#include "../DicomAssoc.h"
#include "../AbstractService.h"

using namespace M4D::Dicom;

///////////////////////////////////////////////////////////////////////

// static members definitions
size_t 
AbstractService::m_numOfInstances;

T_DIMSE_BlockingMode
AbstractService::m_mode;

const size_t
AbstractService::m_maxPDU = 16384;

///////////////////////////////////////////////////////////////////////

AbstractService::AbstractService()
{
	// if it is the first instance let's init network
	if( m_numOfInstances == 0)
	{
		/*
		** Don't let dcmdata remove tailing blank padding or perform other
		** maipulations.  We want to see the real data.
		*/
		dcmEnableAutomaticInputDataCorrection.set(OFFalse);

		/* make sure data dictionary is loaded */
		/*if (!dcmDataDict.isDictionaryLoaded()) {
			fprintf(stderr, "Warning: no data dictionary loaded, check environment variable: %s\n",
					DCM_DICT_ENVIRONMENT_VARIABLE);
		}*/

	#ifdef HAVE_WINSOCK_H
		WSAData winSockData;
		/* we need at least version 1.1 */
		WORD winSockVersionNeeded = MAKEWORD( 1, 1 );
		WSAStartup(winSockVersionNeeded, &winSockData);
	#endif
	}

	m_numOfInstances++;

  m_mode = DIMSE_BLOCKING;
}

///////////////////////////////////////////////////////////////////////

AbstractService::~AbstractService()
{
	// if it is the last instance let's clean up network
	if( m_numOfInstances == 1)
	{
	#ifdef HAVE_WINSOCK_H
		WSACleanup();
	#endif		
	}

	m_numOfInstances--;

	/* drop the network, i.e. free memory of T_ASC_Network* structure. This call */
	/* is the counterpart of ASC_initializeNetwork(...) which was called above. */
	if( ASC_dropNetwork(&m_net).bad() )
	{
		D_PRINT( "Drop network failed!");
	}

}

///////////////////////////////////////////////////////////////////////

/** @} */

