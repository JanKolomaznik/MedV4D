#ifndef M4D_DICOM_STREAM_DECODER
#define M4D_DICOM_STREAM_DECODER

/**
 *  @ingroup dicom
 *  @file dicomDataStreamDecoder.h
 *  @author Vaclav Klecanda
 *  @{
 */

namespace M4D
{
namespace Dicom
{

template<typename T>
std::string PrintBinary(const T &val)
{
	std::stringstream s;
	T tmp = val;
	
	for(unsigned int i=0; i<(sizeof(T)*8); i++)
	{
		s << (tmp & 1);	// print last digit
		tmp = tmp >> 1;		// bitshift to right
	}
	
	// now we have printed the number from back to front
	// so inverse the stream content and output it
	std::string printedString = s.str();
	std::stringstream outs;
	for( 	std::string::reverse_iterator it = printedString.rbegin(); 
			it != printedString.rend(); 
			it++)
		outs << *it;
	
	return outs.str();
}

/**
 *  Decodes data from DICOM data stream. See DICOM doc( [ver]_5 AnnexD) for details
 *  about DICOM data stream decoding.
 */
class DicomDataStreamDecoder
{
private:
  uint32 m_currBitsAvail;
  uint32 m_buf;

  uint32 m_bitsAllocated;
  uint32 m_highBit;
  uint32 m_bitsStored;
  bool m_isDataSigned;

  uint32 m_bitsAllocatedMask;

  uint32 m_item;
  uint32 m_pixelCellMask;

  uint16 *m_stream;
public:

  DicomDataStreamDecoder( 
    uint16 bitsAllocated,
    uint16 highBit,
    uint16 bitsStored,
    bool dataSigned,
    uint16 *stream
    )
    : m_currBitsAvail(0),
    m_buf( 0),
    m_bitsAllocated( (uint32)bitsAllocated),
    m_highBit( (uint32)highBit),
    m_bitsStored( (uint32)bitsStored),
    m_isDataSigned( dataSigned),
    m_stream( stream)
  {
    m_bitsAllocatedMask = ((uint32)(-1) >> (32-bitsAllocated));

    // prepare pixel Cell mask
    {
      m_pixelCellMask = ((uint32)(-1)) & m_bitsAllocatedMask;

      uint32 numZerosToRight = highBit + 1 - m_bitsStored;        
      m_pixelCellMask >>= numZerosToRight; // shift to right
      // shift back to get appropriate cnt of zeros
      m_pixelCellMask <<= numZerosToRight;

      uint32 numZerosLeft = (32 - bitsAllocated) +  // padding to 32bit
        bitsAllocated - m_bitsStored - 
         (m_bitsStored - (highBit + 1) );
      // cut off ones on the left side
      m_pixelCellMask <<= numZerosLeft;
      m_pixelCellMask >>= numZerosLeft;
    }
    
    D_COMMAND( std::cout << "DicomDataStreamDecoder: bitsAlloc: " 
        		<< m_bitsAllocated
        		<< " bitsStored: " 
        		<< m_bitsStored 
        		<< " highBit: " 
        		<< m_highBit 
        		<< " DataSigned: " 
        		<< m_isDataSigned 
        		<< std::endl);
  }

  template< typename T>
  T GetItem( void)
  {
    if( m_currBitsAvail < m_bitsAllocated)
    {
      // load next item from stream
      m_buf += ( ((uint32)*m_stream) << m_currBitsAvail);
      m_currBitsAvail += 16;
      m_stream++;
      
      //      D_COMMAND( std::cout << "CurBuf: " 
      //    		  << PrintBinary<uint64>(m_buf) << std::endl);
    }

    m_item = m_buf & m_bitsAllocatedMask;   // get the val
    m_buf >>= m_bitsAllocated;              // cut off the val from buf
    m_currBitsAvail -= m_bitsAllocated;     // decrease avail bits in buff
    
    //    D_COMMAND( std::cout << "Gettin': " 
    //    		<< PrintBinary<T>((m_item & m_pixelCellMask)) << std::endl);

    // get the pixel value from pixel cell
    return (T) (m_item & m_pixelCellMask);
      
  }

};

} // namespace
}
/** @} */
#endif

