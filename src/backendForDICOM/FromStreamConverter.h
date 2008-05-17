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
  uint8 m_currBitsAvail;
  uint32 m_buf;

  uint8 m_bitsAllocated;
  uint8 m_highBit;
  uint8 m_bitsStored;

  uint32 m_bitsAllocatedMask;

  T m_item;
  T m_pixelCellMask;

  uint16 *m_stream;
public:
  FromStreamConverter( 
    uint8 bitsAllocated,
    uint8 highBit,
    uint8 bitsStored,
    uint16 *stream
    )
    : m_bitsAllocated( bitsAllocated),
    m_highBit( highBit),
    m_bitsStored( bitsStored),
    m_stream( stream)
  {
    m_bitsAllocatedMask = ((uint32)(-1) >> (32-bitsAllocated));

    // prepare pixel Cell mask
    {
      m_pixelCellMask = ((T)(-1)) & m_bitsAllocatedMask;

      uint8 numZerosToRight = bitsAllocated - m_bitsStored - 
         (m_bitsStored - (highBit + 1) );
        
      m_pixelCellMask >>= numZerosToRight; // shift to right
      // shift back to get appropriate cnt of zeros
      m_pixelCellMask <<= numZerosToRight;

      uint numZerosLeft = highBit + 1 - m_bitsStored;
      // cut off ones on the left side
      m_pixelCellMask <<= numZerosLeft;
      m_pixelCellMask >>= numZerosLeft;
    }
  }

  uint8 GetItem( void)
  {
    if( m_currBitsAvail < m_bitsAllocated)
    {
      // load next item from stream
      m_buf += ( ((uint32)*m_stream) << m_currBitsAvail);
      m_currBitsAvail += 16;
    }

    m_item = m_buf & m_bitsAllocatedMask;   // get the val
    m_buf >>= m_bitsAllocated;              // cut off the val from buf
    m_currBitsAvail -= m_bitsAllocated;

    // get the pixel value from pixel cell
    return m_item & m_pixelCellMask;
  }
};

} // namespace
}
#endif