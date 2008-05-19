#ifndef M4D_FROMSTREAMCONVERTER
#define M4D_FROMSTREAMCONVERTER

namespace M4D
{
namespace DicomInternal {
/**
 *  Convertor from DICOM data stream. See DICOM doc( [ver]_5 AnnexD)
 */
template< typename T>
class FromStreamConverter
{
private:
  uint32 m_currBitsAvail;
  uint32 m_buf;

  uint32 m_bitsAllocated;
  uint32 m_highBit;
  uint32 m_bitsStored;

  uint32 m_bitsAllocatedMask;

  T m_item;
  T m_pixelCellMask;

  uint16 *m_stream;
public:
  FromStreamConverter( 
    uint16 bitsAllocated,
    uint16 highBit,
    uint16 bitsStored,
    uint16 *stream
    )
    : m_bitsAllocated( (uint32)bitsAllocated),
    m_highBit( (uint32)highBit),
    m_bitsStored( (uint32)bitsStored),
    m_stream( stream)
  {
    m_bitsAllocatedMask = ((uint32)(-1) >> (32-bitsAllocated));

    // prepare pixel Cell mask
    {
      m_pixelCellMask = (T) (((T)(-1)) & m_bitsAllocatedMask);

      uint32 numZerosToRight = bitsAllocated - m_bitsStored - 
         (m_bitsStored - (highBit + 1) );
        
      m_pixelCellMask >>= numZerosToRight; // shift to right
      // shift back to get appropriate cnt of zeros
      m_pixelCellMask <<= numZerosToRight;

      uint32 numZerosLeft = highBit + 1 - m_bitsStored;
      // cut off ones on the left side
      m_pixelCellMask <<= numZerosLeft;
      m_pixelCellMask >>= numZerosLeft;
    }
  }

  T GetItem( void)
  {
    if( m_currBitsAvail < m_bitsAllocated)
    {
      // load next item from stream
      m_buf += ( ((uint32)*m_stream) << m_currBitsAvail);
      m_currBitsAvail += 16;
    }

    m_item = (T) (m_buf & m_bitsAllocatedMask);   // get the val
    m_buf >>= m_bitsAllocated;              // cut off the val from buf
    m_currBitsAvail -= m_bitsAllocated;

    // get the pixel value from pixel cell
    return (T) (m_item & m_pixelCellMask);
  }
};

} // namespace
}
#endif