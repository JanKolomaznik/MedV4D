#ifndef NEIGHBOURHOODITERATOR_H_
#define NEIGHBOURHOODITERATOR_H_

#include "neighborhoodCell.h"

namespace M4D {
namespace Cell {

template<typename TPixel, uint8 Dimension>
class NeighbourIteratorCell
{
public:
	  typedef TPixel         PixelType;
	  
	  static const uint8 Dim = Dimension;
	  
	  /** Standard class typedefs. */
	  typedef NeighbourIteratorCell<TPixel, Dim>              Self;
	  typedef NeighborhoodCell<TPixel, Dim>		NeighborhoodType;

	  /** Inherit typedefs from superclass */
	  typedef typename NeighborhoodType::TOffset      OffsetType;
	  typedef typename NeighborhoodType::TIndex     IndexType;
	  typedef typename NeighborhoodType::ContinuousIndexType ContinuousIndexType;
	  
	  typedef uint32 OffsetValueType;
	  typedef typename NeighborhoodType::RadiusType      RadiusType;  
	  typedef typename NeighborhoodType::TSize       SizeType;
	  typedef typename NeighborhoodType::TRegion RegionType;
	  typedef typename NeighborhoodType::TStrides StrideType;
	  //typedef typename NeighborhoodType::SizeValueType   SizeValueType;
	  //typedef typename NeighborhoodType::Iterator        Iterator;
	  //typedef typename NeighborhoodType::ConstIterator   ConstIterator;
	  
	  /** Typedef support for common objects */
	  //typedef typename IndexType::IndexValueType       IndexValueType;

	  /** Virtual destructor */
	  virtual ~NeighbourIteratorCell() {}
	  
	  NeighborhoodType &GetNeighborhood() const { return *m_neighbourhood; }

	  /** Constructor which establishes the region size, neighborhood, and image
	   * over which to walk. */
	  NeighbourIteratorCell() : m_neighbourhood(0) {}
	  NeighbourIteratorCell(NeighborhoodType *neiborhood)
	  	: m_neighbourhood(neiborhood)
	    {
	    //this->Initialize(radius, ptr, region);
	    for (unsigned int i=0; i < Dimension; i++)
	      { m_InBounds[i] = false; }
	    }
	  
	  void SetNeighbourhood(NeighborhoodType *neiborhood)
	  {
		  m_neighbourhood = neiborhood;  
	  }

	  /** Computes the internal, N-d offset of a pixel array position n from 
	   * (0,0, ..., 0) in the "upper-left" corner of the neighborhood. */
	  OffsetType ComputeInternalIndex(unsigned int n) const;

//	  /** Returns the array of upper loop bounds used during iteration. */
//	  IndexType GetBound() const
//	    {    return m_Bound;   }
//
//	  /** Returns the loop bound used to define the edge of a single
//	   * dimension in the itk::Image region. */
//	  long GetBound(unsigned int n) const
//	    {    return m_Bound[n];  }
	  
	  /** Returns the pointer to the center pixel of the neighborhood. */
	  const TPixel *GetCenterPointer() const
	    {    return m_neighbourhood->GetPixel(this->Size()>>1);  }
	  
	  /** Returns the pixel referenced at the center of the 
	   *  ConstNeighborhoodIterator. */
	  PixelType GetCenterPixel() const
	    {return m_neighbourhood->GetPixel( m_neighbourhood->GetCenterNeighborhoodIndex() );}
	  
//	  /** Virtual function that "dereferences" a ConstNeighborhoodIterator,
//	   * returning a Neighborhood of pixel values. */
//	  virtual NeighborhoodType GetNeighborhood() const;
	  
	  PixelType GetPixel(const unsigned i) const { return m_neighbourhood->GetPixel( i ); }

	  /** Returns the pixel value located at a linear array location i. */
	  virtual PixelType GetPixel(const unsigned i,bool& IsInBounds) const
	    { 
	    if( !m_NeedToUseBoundaryCondition )
	      {
	      return m_neighbourhood->GetPixel( i );
	      }
	    IsInBounds = true;
	   // bool inbounds; 
	    return m_neighbourhood->GetPixel( i ); 
	    }

	  /** Return the pixel value located at a linear array location i.
	   * Sets "IsInBounds" to true if the location is inside the
	   * image and the pixel value returned is an actual pixel in the
	   * image. Sets "IsInBounds" to false if the location is outside the
	   * image and the pixel value returned is a boundary condition. */
	  //virtual PixelType GetPixel(const unsigned i, bool& IsInBounds) const;

	  /** Returns the pixel value located at the itk::Offset o from the center of
	      the neighborhood. */
	  virtual PixelType GetPixel(const OffsetType &o) const
	    { 
	    bool inbounds; 
	    return (this->GetPixel(m_neighbourhood->GetNeighborhoodIndex(o), inbounds)); 
	    }
	  
	  

	  /** Returns the pixel value located at the itk::Offset o from the center of
	   * the neighborhood. Sets "IsInBounds" to true if the offset is inside the
	   * image and the pixel value returned is an actual pixel in the
	   * image. Sets "IsInBounds" to false if the offset is outside the
	   * image and the pixel value returned is a boundary condition. */
	  virtual PixelType GetPixel(const OffsetType &o,
	                             bool& IsInBounds) const
	    {return (this->GetPixel(m_neighbourhood->GetNeighborhoodIndex(o), IsInBounds)); }
	  
	  /** Returns the pixel value located i pixels distant from the neighborhood 
	   *  center in the positive specified ``axis'' direction. No bounds checking 
	   *  is done on the size of the neighborhood. */
	  virtual PixelType GetNext(const unsigned axis, const unsigned i) const
	    { return (GetPixel(m_neighbourhood->GetCenterNeighborhoodIndex()
	                           + (i * m_neighbourhood->GetStride(axis)))); }

	  /** Returns the pixel value located one pixel distant from the neighborhood
	   *  center in the specifed positive axis direction. No bounds checking is 
	   *  done on the size of the neighborhood. */
	  virtual PixelType GetNext(const unsigned axis) const
	    { return (GetPixel(m_neighbourhood->GetCenterNeighborhoodIndex()
	                           + m_neighbourhood->GetStride(axis))); }

	  /** Returns the pixel value located i pixels distant from the neighborhood 
	   *  center in the negative specified ``axis'' direction. No bounds checking 
	   *  is done on the size of the neighborhood. */
	  virtual PixelType GetPrevious(const unsigned axis, const unsigned i) const
	    { return (GetPixel(m_neighbourhood->GetCenterNeighborhoodIndex()
	                           - (i * m_neighbourhood->GetStride(axis)))); }
	  
	  /** Returns the pixel value located one pixel distant from the neighborhood 
	   *  center in the specifed negative axis direction. No bounds checking is 
	   *  done on the size of the neighborhood. */
	  virtual PixelType GetPrevious(const unsigned axis) const
	    { return (GetPixel(m_neighbourhood->GetCenterNeighborhoodIndex()
	                           - m_neighbourhood->GetStride(axis))); } 
	  
	  /** Returns the N-dimensional index of the iterator's position in
	     * the image. */
	  IndexType GetIndex(void) const
	      { return m_Loop;  }
	  
	  /** Returns the image index for neighbor pixel at offset o from the center of
	      the neighborhood. */
	  virtual IndexType GetIndex(const OffsetType &o) const
	  { return (this->GetIndex() + o); }

	  /** Returns the image index for neighbor pixel at index i in the
	      neighborhood. */
	  virtual IndexType GetIndex(const unsigned i) const
	  { return (this->GetIndex() + this->GetOffset(i)); }
	  
	  /**  Returns the region of iteration. */
	  RegionType GetRegion() const
	    { return m_Region; }
	  
	  /** Returns the N-dimensional starting index of the iterator's position on
	   * the image. */
	  IndexType GetBeginIndex() const
	    { return m_BeginIndex; }

	  /** Returns a bounding box for the region spanned by this neighborhood
	      represented by an itk::ImageRegion */
	  RegionType GetBoundingBoxAsImageRegion() const;
	  
	  /** Returns the offsets used to wrap across dimensional boundaries. */
	  OffsetType GetWrapOffset() const
	    {  return m_WrapOffset;  }

//	  /** Returns the internal offset associated with wrapping around a single
//	   * dimension's region boundary in the itk::Image.  An offset for each
//	   * dimension is necessary to shift pointers when wrapping around region
//	   * edges because region memory is not necessarily contiguous within the
//	   * buffer. */
//	  OffsetValueType GetWrapOffset(unsigned int n) const
//	    {    return m_WrapOffset[n];   }

	  /** Virtual method for rewinding the iterator to its beginning pixel.
	   * This is useful for writing functions which take neighborhood iterators
	   * of arbitrary type and must use virtual functions. */
	  virtual void GoToBegin();
	  
	  /** Virtual method for sending the iterator to one past the last pixel in its
	   * region. */
	  virtual void GoToEnd();
	  
//	  /** Initializes the iterator to walk a particular image and a particular
//	   * region of that image. */
//	  virtual void Initialize(const SizeType &radius, const ImageType *ptr,
//	                          const RegionType &region);
//
//	  /** Virtual method for determining whether the iterator is at the
//	   * beginning of its iteration region. */
//	  virtual bool IsAtBegin() const
//	    {    return ( this->GetCenterPointer() == m_Begin );   }
//	  
//	  /** Virtual method for determining whether the iterator has reached the
//	   * end of its iteration region. */
//	  virtual bool IsAtEnd() const
//	    {
//	    if ( this->GetCenterPointer() > m_End )
//	      {
//	      ExceptionObject e(__FILE__, __LINE__);
//	      OStringStream msg;
//	      msg << "In method IsAtEnd, CenterPointer = " << this->GetCenterPointer()
//	          << " is greater than End = " << m_End
//	          << std::endl
//	          << "  " << *this;
//	      e.SetDescription(msg.str().c_str());
//	      throw e;
//	      }
//	    return ( this->GetCenterPointer() == m_End );
//	    }
//	  
	  /** Increments the pointers in the ConstNeighborhoodIterator,
	   * wraps across boundaries automatically, accounting for
	   * the disparity in the buffer size and the region size of the
	   * image. */
	  Self &operator++();
	  
	  /** Decrements the pointers in the ConstNeighborhoodIterator,
	   * wraps across boundaries automatically, accounting for
	   * the disparity in the buffer size and the region size of the
	   * image. */
	  Self &operator--();  
	 
	  /** Returns a boolean == comparison of the memory addresses of the center
	   * elements of two ConstNeighborhoodIterators of like pixel type and
	   * dimensionality.  The radii of the iterators are ignored. */
	  bool operator==(const Self &it) const 
	    {   return  it.GetCenterPointer() == this->GetCenterPointer();   }
	  
	  /** Returns a boolean != comparison of the memory addresses of the center
	   * elements of two ConstNeighborhoodIterators of like pixel type and
	   * dimensionality.  The radii of the iterators are ignored. */
	  bool operator!=(const Self &it) const
	    {    return  it.GetCenterPointer() != this->GetCenterPointer();  }
	  
	  /** Returns a boolean < comparison of the memory addresses of the center
	   * elements of two ConstNeighborhoodIterators of like pixel type and
	   * dimensionality.  The radii of the iterators are ignored. */
	  bool operator<(const Self &it) const
	    {  return  this->GetCenterPointer() < it.GetCenterPointer();  }

	  /** Returns a boolean < comparison of the memory addresses of the center
	   * elements of two ConstNeighborhoodIterators of like pixel type and
	   * dimensionality.  The radii of the iterators are ignored. */
	  bool operator<=(const Self &it) const
	    {    return  this->GetCenterPointer() <= it.GetCenterPointer();  }
	  
	  /** Returns a boolean > comparison of the memory addresses of the center
	   * elements of two ConstNeighborhoodIterators of like pixel type and
	   * dimensionality.  The radii of the iterators are ignored. */
	  bool operator>(const Self &it) const
	    {    return  this->GetCenterPointer() > it.GetCenterPointer();  }

	  /** Returns a boolean >= comparison of the memory addresses of the center
	   * elements of two ConstNeighborhoodIterators of like pixel type and
	   * dimensionality.  The radii of the iterators are ignored. */
	  bool operator>=(const Self &it) const
	    {    return  this->GetCenterPointer() >= it.GetCenterPointer();  }

	  /** This method positions the iterator at an indexed location in the
	   * image. SetLocation should _NOT_ be used to update the position of the
	   * iterator during iteration, only for initializing it to a position
	   * prior to iteration.  This method is not optimized for speed. */
	  void SetLocation( const IndexType& position )
	    {
		  m_neighbourhood->SetPosition(position);
	    }
	  

//	  /** Addition of an itk::Offset.  Note that this method does not do any bounds
//	   * checking.  Adding an offset that moves the iterator out of its assigned
//	   * region will produce undefined results. */
//	  Self &operator+=(const OffsetType &);
//
//	  /** Subtraction of an itk::Offset. Note that this method does not do any 
//	   *  bounds checking.  Subtracting an offset that moves the iterator out 
//	   * of its assigned region will produce undefined results. */
//	  Self &operator-=(const OffsetType &);
//
//	  /** Distance between two iterators */
//	  OffsetType operator-(const Self& b)
//	    {  return m_Loop - b.m_Loop;  }

	  /** Returns false if the iterator overlaps region boundaries, true
	   * otherwise.  Also updates an internal boolean array indicating
	   * which of the iterator's faces are out of bounds. */
	  bool InBounds() const;
	  
	  OffsetType GetOffset(unsigned int i) const
	    {    return m_OffsetTable[i];  }
	  
	protected:
		
		void ComputeNeighborhoodOffsetTable();
	  
	  /** Virtual method for setting internal loop boundaries.  This
	   * method must be defined in each subclass because
	   * each subclass may handle loop boundaries differently. */
//	  virtual void SetBound(const SizeType &);


	  /** Default method for setting the index of the first pixel in the
	   * iteration region. */
	  virtual void SetBeginIndex( const IndexType& start)
	    {  m_BeginIndex = start;  }

	  /** Default method for setting the index of the first pixel in the
	   * iteration region. */
	  //virtual void SetEndIndex();
	  
	  IndexType *m_OffsetTable;
	  
	  /** The starting index for iteration within the itk::Image region
	   * on which this ConstNeighborhoodIterator is defined. */
	  IndexType m_BeginIndex;

	  /** An array of upper looping boundaries used during iteration. */
	  IndexType m_Bound;

	  /** A pointer to the first pixel in the iteration region. */
	  const TPixel *m_Begin;

	  /** A pointer to one past the last pixel in the iteration region. */
	  const TPixel *m_End;

	  /** The end index for iteration within the itk::Image region
	   * on which this ConstNeighborhoodIterator is defined. */
	  IndexType m_EndIndex;

	  /** Array of loop counters used during iteration. */
	  IndexType m_Loop;
	 
	  /** The region over which iteration is defined. */
	  RegionType m_Region;

	  /** The internal array of offsets that provide support for regions of 
	   *  interest.
	   *  An offset for each dimension is necessary to shift pointers when wrapping
	   *  around region edges because region memory is not necessarily contiguous
	   *  within the buffer. */
	  OffsetType m_WrapOffset;


	  /** Denotes which of the iterators dimensional sides spill outside
	   * region of interest boundaries. */
	  mutable bool m_InBounds[Dimension];

	  /** Denotes if iterator is entirely within bounds */
	  mutable bool m_IsInBounds;
	  
	  /** Is the m_InBounds and m_IsInBounds variables up to date? Set to
	   * false whenever the iterator is repositioned.  Set to true within
	   * InBounds(). */
	  mutable bool m_IsInBoundsValid;
	  
	  bool m_NeedToUseBoundaryCondition;
	  
	  /** Lower threshold of in-bounds loop counter values. */
	  IndexType m_InnerBoundsLow;
	  
	  /** Upper threshold of in-bounds loop counter values. */
	  IndexType m_InnerBoundsHigh;

	  NeighborhoodType *m_neighbourhood;
};

//include implementation
#include "src/neighbourhoodIterator.tcc"

}}  // namespace

#endif /*NEIGHBOURHOODITERATOR_H_*/