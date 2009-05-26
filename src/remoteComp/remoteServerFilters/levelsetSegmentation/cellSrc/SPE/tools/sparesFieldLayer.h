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
  
#define Address2Node(add) ((NodeType *)add.Get64())
  
  class Iterator
  {
  public:
	  bool HasNext() {
		  return Address2Node(_it->Next) != _end;
	  }
	  
	  NodeType *Next() { 
		  NodeType * _tmp = _it;
		  _it = Address2Node(_it->Next);
		  return _tmp;
	  }
	  NodeType *_it;
	  NodeType *_end;
  };
  

  
  void InitIterator(Iterator &it)
  {
	  it._it = Front();
	  it._end = &m_HeadNode;
  }

  /** Returns a pointer to the first node in the list.  Constant
   * time. */ 
  NodeType *Front()
    { return Address2Node(m_HeadNode.Next); }

  /** Returns a const pointer to the first node in the list. Constant time. */
  const NodeType *Front() const
    { return Address2Node(m_HeadNode.Next); }
  
  const NodeType *End() const
  {
	  return &m_HeadNode;
  }

  /** Unlinks the first node from the list. Constant time. */
  void PopFront()
    {
    m_HeadNode.Next = Address2Node(m_HeadNode.Next)->Next;
    Address2Node(m_HeadNode.Next)->Previous = (uint64)&m_HeadNode;
    m_Size -= 1;
    }
  
  /** Links a node into the front of the list. Constant time. */
  void PushFront(NodeType *n)
    {
    n->Next = m_HeadNode.Next;
    n->Previous = (uint64)&m_HeadNode;
    Address2Node(m_HeadNode.Next)->Previous = (uint64)n;
    m_HeadNode.Next = (uint64)n;
    m_Size += 1;
    }
  
  /** Unlinks a node from the list */
  void Unlink(NodeType *n)
    {
	  Address2Node(n->Previous)->Next = n->Next;
	  Address2Node(n->Next)->Previous = n->Previous;
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
	  m_HeadNode.Next = (uint64)&m_HeadNode;
	  m_HeadNode.Previous = (uint64)&m_HeadNode;
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
