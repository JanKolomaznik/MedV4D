
#include "ImageSerializer.h"
#include "Imaging/Image.h"

using namespace M4D::CellBE;
using namespace M4D::Imaging;

///////////////////////////////////////////////////////////////////////////////

void
ImageSerializer::SerializeProperties(M4D::CellBE::NetStream &s)
{
  AbstractImage *im = (AbstractImage *) m_dataSet; // cast to sucessor

  // serialize common properties
  s << (uint8) im->GetDimension() << (uint8) im->GetElementTypeID();

  // now switch according dimensionality
  switch( im->GetDimension() )
  {
  case 2:
    break;

  case 3:
    break;

  case 4:
    break;
  }
}

///////////////////////////////////////////////////////////////////////////////

M4D::Imaging::AbstractDataSet *
ImageSerializer::DeSerializeProperties(M4D::CellBE::NetStream &s)
{

  uint8 dim, elemType;
  s >> dim >> elemType;   // get common properties

  // construct new Image instance according that common props
  switch( dim)
  {
  case 2:
    break;

  case 3:
    break;

  case 4:
    break;
  }

  return m_dataSet; // TODO initialize

}

///////////////////////////////////////////////////////////////////////////////

void
ImageSerializer::OnDataPieceReadRequest( 
                                      DataPieceHeader *header, DataBuffs &bufs)
{
}

///////////////////////////////////////////////////////////////////////////////

void
ImageSerializer::OnDataSetEndRead( void)
{
}

///////////////////////////////////////////////////////////////////////////////

void
ImageSerializer::Serialize( M4D::CellBE::iPublicJob *job)
{
}

///////////////////////////////////////////////////////////////////////////////

void
ImageSerializer::Reset( void)
{
}

///////////////////////////////////////////////////////////////////////////////