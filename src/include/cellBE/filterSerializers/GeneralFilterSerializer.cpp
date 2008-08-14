
#include <vector>
#include "Common.h"
#include "cellBE/GeneralFilterSerializer.h"

#include "cellBE/filterSerializers/ThresholdingSerializer.h"

using namespace M4D::CellBE;
using namespace M4D::Imaging;
using namespace std;

GeneralFilterSerializer::FilterSerializers
GeneralFilterSerializer::m_filterSerializers;

///////////////////////////////////////////////////////////////////////////////

GeneralFilterSerializer::GeneralFilterSerializer()
{
  // here is to be put each new filterSerializer instance
  m_filterSerializers.insert( FilterSerializers::value_type(
    Thresholding, 
    new FilterSerializer< typename ThresholdingFilter< Image<uint8, 3> >::Properties >(
      NULL )
    ) );
  // ...
}

///////////////////////////////////////////////////////////////////////////////

//AbstractPipeFilter * 
//GeneralFilterSerializer::DeSerialize( M4D::CellBE::NetStream &s)
//{
//  //switch( (FilterID) filterID)
//  //{
//  //case Thresholding:
//  //  fs = new ThresholdingSetting();
//  //  fs->DeSerialize(s);
//  //  m_filters.push_back( fs);
//  //  break;
//
//  //default:
//  //  LOG( "Unrecognized filter");
//  //  throw ExceptionBase("Unrecognized filter");
//  //}
//  return NULL;
//}
//
/////////////////////////////////////////////////////////////////////////////////
//
//template< typename FilterProperties >
//AbstractFilterSerializer *
//GeneralFilterSerializer::GetFilterSerializer( FilterProperties *props )
//{
//  switch( GetFilterID<typename FilterProperties>( props) )
//  {
//    case Thresholding:
//      // return thresholding serializator instance
//      return new ThresholdingSerializer();
//      break;
//
//    default:
//      LOG( "Unrecognized filter");
//      throw ExceptionBase("Unrecognized filter");
//      break;
//  }
//}

///////////////////////////////////////////////////////////////////////////////