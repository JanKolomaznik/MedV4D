#ifndef SPARESFIELDLAYER_H_
#define SPARESFIELDLAYER_H_

namespace M4D {
namespace Cell {

template <class TNodeType>
class SparseFieldLayer
{
public:
  /** Standard typedefs. */
  typedef SparseFieldLayer          Self;

  /** Type of node stored in the linked list. */
  typedef TNodeType NodeType;
  
  /** Alias for the type of value stored in the list. Conforms to Standard Template
   *  Library vocabulary. */
  typedef NodeType ValueType;

  /** Iterator type for the list. */
//  typedef SparseFieldLayerIterator<NodeType> Iterator;
//
//  /** Const iterator type for the list. */
//  typedef ConstSparseFieldLayerIterator<NodeType> ConstIterator;

//  /** Regions used for multithreading */
//  struct RegionType 
//    {
//    ConstIterator first;
//    ConstIterator last;  // this is one past the actual last element
//    };   
//  
//  typedef std::vector<RegionType> RegionListType;

  /** Returns a pointer to the first node in the list.  Constant
   * time. */ 
  NodeType *Front()
    { return m_HeadNode.Next; }

  /** Returns a const pointer to the first node in the list. Constant time. */
  const NodeType *Front() const
    { return m_HeadNode.Next; }

  /** Unlinks the first node from the list. Constant time. */
  void PopFront()
    {
    m_HeadNode.Next = m_HeadNode.Next->Next;
    m_HeadNode.Next->Previous = &m_HeadNode;
    m_Size -= 1;
    }
  
  /** Links a node into the front of the list. Constant time. */
  void PushFront(NodeType *n)
    {
    n->Next = m_HeadNode.Next;
    n->Previous = &m_HeadNode;
    m_HeadNode.Next->Previous = n;
    m_HeadNode.Next = n;
    m_Size += 1;
    }
  
  /** Unlinks a node from the list */
  void Unlink(NodeType *n)
    {
    n->Previous->Next = n->Next;
    n->Next->Previous = n->Previous;
    m_Size -= 1;
    }
  
//  /** Returns an iterator pointing to the first node in the list. */
//  Iterator Begin()
//    { return Iterator(m_HeadNode.Next); }
//
//  /** Returns a const iterator pointing to the first node in the
//   * list. */
//  ConstIterator Begin() const
//    { return ConstIterator(m_HeadNode.Next); }
//
//  /** Returns an iterator pointing one node past the end of the list. */
//  Iterator End()
//    { return Iterator(m_HeadNode); }
//
//  /** Returns a const iterator pointing one node past the end of the list. */
//  ConstIterator End() const
//    { return ConstIterator(m_HeadNode); }
  
  /** Returns TRUE if the list is empty, FALSE otherwise. Executes in constant
   *  time. */
  bool Empty() const
    {
//    if (m_HeadNode.Next == &m_HeadNode) 
//    	return true;
//    else return false;
	  return m_Size == 0;
    }
  
  /** Returns the number of elements in the list. Size() executes in constant
   *  time. */
  unsigned int Size() const { return m_Size; }
  
  /** Returns pointers to first and last+1 elements of num partitions of 
      the itkSparseFieldLayer */
//  RegionListType SplitRegions(int num) const;

  SparseFieldLayer()
  {
	  m_HeadNode.Next = &m_HeadNode;
	  m_HeadNode.Previous = &m_HeadNode;
	  m_Size = 0;
  }
  ~SparseFieldLayer()
  {
  }
  
private:
  SparseFieldLayer(const Self&);    //purposely not implemented
  void operator=(const Self&);      //purposely not implemented
  
  /** The anchor node of the list.  m_HeadNode.Next is the first node in the
   *  list. If m_HeadNode.Next == m_HeadNode, then the list is empty. */
  NodeType   m_HeadNode;
  unsigned int m_Size;
};
    
    }}

#endif /*SPARESFIELDLAYER_H_*/
