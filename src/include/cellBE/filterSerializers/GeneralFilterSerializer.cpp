/**
 *  @ingroup cellbe
 *  @file GeneralFilterSerializer.cpp
 *  @author Vaclav Klecanda
 */
#include <vector>
#include "Common.h"
#include "cellBE/GeneralFilterSerializer.h"

using namespace M4D::CellBE;
using namespace std;

FilterSerializerArray GeneralFilterSerializer::m_filterSerializers;

///////////////////////////////////////////////////////////////////////////////

//GeneralFilterSerializer::GeneralFilterSerializer()
//{
//  
//}

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