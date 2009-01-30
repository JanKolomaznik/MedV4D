/**
 * @ingroup cellbe 
 * @author Vaclav Klecanda 
 * @file GeneralDataSetSerializer.cpp 
 * @{ 
 **/

/**
 *  @ingroup cellbe
 *  @file GeneralDataSetSerializer.cpp
 *  @author Vaclav Klecanda
 */
#include "Common.h"
#include "cellBE/GeneralDataSetSerializer.h"

// includes of particular dataSet Serializers
#include "cellBE/dataSetSerializers/ImageSerializer.h"

using namespace M4D::Imaging;
using namespace std;

namespace M4D {
namespace CellBE {

///////////////////////////////////////////////////////////////////////////////

AbstractDataSetSerializer *
GeneralDataSetSerializer::GetDataSetSerializer( 
  M4D::Imaging::AbstractDataSet *dataSet)
{
  switch( dataSet->GetDatasetType() )
  {
  case DATASET_IMAGE:    
    NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO( 
      ((AbstractImage *)dataSet)->GetElementTypeID(),
		  DIMENSION_TEMPLATE_SWITCH_MACRO( 
        ((AbstractImage *)dataSet)->GetDimension(),
			  return new ImageSerializer< TTYPE, DIM >(dataSet) )
		  );
    break;

  case DATASET_TRIANGLE_MESH:
    return NULL;
    // TOBEDONE LATERON
    break;
  
  default:
    throw WrongDSetException();
  }
  return NULL;
}

///////////////////////////////////////////////////////////////////////////////

void
GeneralDataSetSerializer::DeSerializeDataSetProperties( 
      AbstractDataSetSerializer **dataSetSerializer
      , M4D::Imaging::AbstractDataSet::ADataSetPtr *returnedDataSet
      , M4D::CellBE::NetStream &s)
{
  uint8 type;
  s >> type;

	switch( (DataSetType) type)
	{
	case DATASET_IMAGE:
		uint16 dim, elemType;
		s >> dim >> elemType;   // get common properties

		if( *dataSetSerializer == NULL)
		{
		NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO( elemType, 
		    DIMENSION_TEMPLATE_SWITCH_MACRO( dim, 
			    *dataSetSerializer = new ImageSerializer< TTYPE, DIM >() )
		  );
		}

		*returnedDataSet = (*dataSetSerializer)->DeSerializeProperties( s);
		(*dataSetSerializer)->SetDataSet( & (*returnedDataSet->get()) );
		break;
	default:
		throw M4D::ErrorHandling::ETODO();

	}
}

///////////////////////////////////////////////////////////////////////////////

}
}

/** @} */

