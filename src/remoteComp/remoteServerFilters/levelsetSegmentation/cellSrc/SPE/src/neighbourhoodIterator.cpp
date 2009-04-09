
#include "common/Types.h"
#include "../neighbourhoodIterator.h"

using namespace M4D::Cell;

///////////////////////////////////////////////////////////////////////////////


NeighbourIteratorCell
::NeighbourIteratorCell()
	: m_neighbourhood(0) 
{
}

///////////////////////////////////////////////////////////////////////////////


NeighbourIteratorCell
::NeighbourIteratorCell(NeighborhoodType *neiborhood)
	: m_neighbourhood(neiborhood)
{
//this->Initialize(radius, ptr, region);
for (unsigned int i=0; i < DIM; i++)
  { m_InBounds[i] = false; }
}

///////////////////////////////////////////////////////////////////////////////


NeighbourIteratorCell
::~NeighbourIteratorCell()
{
	
}

///////////////////////////////////////////////////////////////////////////////


bool
NeighbourIteratorCell
::InBounds() const
{ 
  bool ans = true;
  for (unsigned int i=0; i<DIM; i++)
    {
    if (m_Loop[i] < m_InnerBoundsLow[i] || m_Loop[i] >= m_InnerBoundsHigh[i])
      {
      m_InBounds[i] = ans = false;
      }
    else
      {
      m_InBounds[i] = true;
      }
    }
  m_IsInBounds = ans;
  m_IsInBoundsValid = true;
  return ans;
}


void
NeighbourIteratorCell::ComputeNeighborhoodOffsetTable()
{
  TOffset o;
  unsigned int i, j;
  for (j = 0; j < DIM; j++)
    {
    o[j] = -(static_cast<long>(m_neighbourhood->GetStride(j)));
    }

  for (i = 0; i < m_neighbourhood->GetSize(); ++i)
    {
	  m_OffsetTable[i] = o;
    for (j= 0; j< DIM; j++)
      {
      o[j] = o[j] + 1;
      if (o[j] > static_cast<long>(m_neighbourhood->GetStride(j)))
        {
        o[j] = -(static_cast<long>(m_neighbourhood->GetStride(j)));
        }
      else break;
      }
    }
} 


//
//typename NeighbourIteratorCell::PixelType
//NeighbourIteratorCell
//::GetPixel(const unsigned n, bool& IsInBounds) const
//{
//  // If the region the iterator is walking (padded by the neighborhood size)
//  // never bumps up against the bounds of the buffered region, then don't
//  // bother checking any boundary conditions
//  if (!m_NeedToUseBoundaryCondition)
//    {
//    IsInBounds = true;
//    return (m_neighbourhood->GetPixel(n));
//    }
//
//  register unsigned int i;
//  OffsetValueType OverlapLow, OverlapHigh;
//  OffsetType temp, offset;
//  bool flag;
//
//  // Is this whole neighborhood in bounds?
//  if (this->InBounds())
//    {
//    IsInBounds = true;
//    return (m_neighbourhood->GetPixel(n));
//    }
//  else
//    {
//    temp = this->ComputeInternalIndex(n);
//      
//    flag = true;
//
//    // Is this pixel in bounds?
//    for (i=0; i<Dim; ++i)
//      {
//      if (m_InBounds[i])
//        {
//        offset[i] = 0; // this dimension in bounds
//        }
//      else  // part of this dimension spills out of bounds
//        {
//        // Calculate overlap for this dimension
//        OverlapLow = m_InnerBoundsLow[i] - m_Loop[i];
//        OverlapHigh =
//          static_cast<OffsetValueType>(m_neighbourhood->GetSize(i) - 
//                                     ( (m_Loop[i]+2) - m_InnerBoundsHigh[i] ));
//
//        // 
//        if (temp[i] < OverlapLow)
//          {
//          flag = false;
//          offset[i] = OverlapLow - temp[i];
//          }
//        else if ( OverlapHigh < temp[i] )
//          {
//          flag = false;
//          offset[i] =  OverlapHigh - temp[i];
//          }
//        else offset[i] = 0;
//        }
//      }
//
//    if (flag) 
//      {
//      IsInBounds = true;
//      return m_neighbourhood->GetPixel(n);
//      }
//    else 
//      {
//      IsInBounds = false;
//      return( m_neighbourhood.m_neighbourhoodGetPixel(offset) );
//      }
//    } 
//}



TOffset
NeighbourIteratorCell
::ComputeInternalIndex(unsigned int n) const
{
  TOffset ans;
  long D = (long)DIM;
  unsigned long r;
  r = (unsigned long)n;
  for (long i = D-1; i >= 0; --i)
    {
    ans[i] = static_cast<TOffset::TValue>(r / m_neighbourhood->GetStride(i));
    r = r % m_neighbourhood->GetStride(i);
    }
  return ans;
}



//TRegion
//NeighbourIteratorCell
//::GetBoundingBoxAsImageRegion() const
//{
//  TRegion ans;
//  uint32 zero = 0;
//  ans.offset = (this->GetIndex(zero));
//  ans.size = (this->GetSize());
//  
//  return ans;
//}

//
//NeighbourIteratorCell
//::NeighbourIteratorCell()
//{
//  IndexType zeroIndex; zeroIndex.Fill(0);
//  SizeType  zeroSize; zeroSize.Fill(0);
//
//  m_Bound.Fill(0);
//  m_Begin = 0;
//  m_BeginIndex.Fill(0);
//  // m_ConstImage
//  m_End   = 0;
//  m_EndIndex.Fill(0);
//  m_Loop.Fill(0);
//  m_Region.SetIndex(zeroIndex);
//  m_Region.SetSize(zeroSize);
//  
//  m_WrapOffset.Fill(0);
//
//  for (unsigned int i=0; i < Dimension; i++)
//    { m_InBounds[i] = false; }
//
//  m_IsInBounds = false;
//  m_IsInBoundsValid = false;
//}

//
//NeighbourIteratorCell
//::ConstNeighborhoodIterator(const Self& orig)
//  : Neighborhood<InternalPixelType *, Dimension>(orig)
//{
//  m_Bound      = orig.m_Bound;
//  m_Begin      = orig.m_Begin;
//  m_BeginIndex = orig.m_BeginIndex;  
//  m_ConstImage = orig.m_ConstImage;
//  m_End        = orig.m_End;
//  m_EndIndex   = orig.m_EndIndex;
//  m_Loop       = orig.m_Loop;
//  m_Region     = orig.m_Region;
//  m_WrapOffset = orig.m_WrapOffset;
//
//  m_InternalBoundaryCondition = orig.m_InternalBoundaryCondition;
//  m_NeedToUseBoundaryCondition = orig.m_NeedToUseBoundaryCondition;
//  for (unsigned int i = 0; i < Dimension; ++i)
//    {
//    m_InBounds[i] = orig.m_InBounds[i];
//    }
//  m_IsInBoundsValid = orig.m_IsInBoundsValid;
//  m_IsInBounds = orig.m_IsInBounds;
//
//  m_InnerBoundsLow  = orig.m_InnerBoundsLow;
//  m_InnerBoundsHigh = orig.m_InnerBoundsHigh;
//
//  // Check to see if the default boundary
//  // conditions have been overridden.
//  if ( orig.m_BoundaryCondition ==
//       static_cast<ImageBoundaryConditionConstPointerType>(
//                                          &orig.m_InternalBoundaryCondition ))
//    {
//    this->ResetBoundaryCondition();
//    }
//  else 
//    { m_BoundaryCondition = orig.m_BoundaryCondition; }
//
//  m_NeighborhoodAccessorFunctor = orig.m_NeighborhoodAccessorFunctor;
//  
//}

//
//void
//NeighbourIteratorCell
//::SetEndIndex()
//{
//  if (m_Region.GetNumberOfPixels() > 0)
//    {
//    m_EndIndex = m_Region.GetIndex();
//    m_EndIndex[Dimension-1] = m_Region.GetIndex()[Dimension-1] +
//      static_cast<long>(m_Region.GetSize()[Dimension-1]);
//    }
//  else
//    {
//    // Region has no pixels, so set the end index to be the begin index
//    m_EndIndex = m_Region.GetIndex();
//    }
//}


void
NeighbourIteratorCell
::GoToBegin()
{
  this->SetLocation( m_BeginIndex );
}


void
NeighbourIteratorCell
::GoToEnd()
{
  this->SetLocation( m_EndIndex );
}

//
//NeighbourIteratorCell
//::Initialize(const SizeType &radius, const ImageType *ptr,
//             const RegionType &region)
//{
//  const IndexType regionIndex = region.GetIndex();
//
//  this->SetBeginIndex(region.GetIndex());
//  this->SetLocation(region.GetIndex());
//  this->SetBound(region.GetSize());
//  this->SetEndIndex();
//  
//  m_Begin = ptr->GetBufferPointer() + ptr->ComputeOffset(regionIndex);
//
//  m_End = ptr->GetBufferPointer() + ptr->ComputeOffset( m_EndIndex );
//
//}

//
//void NeighbourIteratorCell
//::SetBound(const SizeType& size)
//{
//  SizeType radius  = m_neighbourhood->GetRadius();
//  const OffsetValueType *offset   = m_ConstImage->GetOffsetTable();
//  const IndexType imageBRStart  = m_ConstImage->GetBufferedRegion().GetIndex();
//  SizeType imageBRSize = m_ConstImage->GetBufferedRegion().GetSize();
//
//  // Set the bounds and the wrapping offsets. Inner bounds are the loop
//  // indicies where the iterator will begin to overlap the edge of the image
//  // buffered region.
//  for (unsigned int i=0; i<Dimension; ++i)
//    {
//    m_Bound[i] = m_BeginIndex[i] + static_cast<IndexValueType>(size[i]);
//    m_InnerBoundsHigh[i]= static_cast<IndexValueType>(imageBRStart[i] 
//                + ( imageBRSize[i]) - static_cast<SizeValueType>(radius[i]) );
//    m_InnerBoundsLow[i] = static_cast<IndexValueType>(imageBRStart[i] 
//                                                                   + radius[i]);
//    m_WrapOffset[i]     = (static_cast<OffsetValueType>(imageBRSize[i]) 
//                                - ( m_Bound[i] - m_BeginIndex[i] )) * offset[i];
//    }
//  m_WrapOffset[Dimension-1] = 0; // last offset is zero because there are no
//                                 // higher dimensions  
//}

