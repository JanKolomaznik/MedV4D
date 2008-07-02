#ifndef RESOURCE_POOL_H
#define RESOURCE_POOL_H

#include <stack>

template<class T, uint16 size>
class Pool
{
  typedef std::stack< T *> FreeStack;

  T m_array[size];
  FreeStack m_freeItems;

public:
  Pool()
  {
    // all are free at the beginning
    for( int i=0; i<size; i++)
      m_freeItems.push( &m_array[i]);
  }

  T *GetFreeItem( void) 
  { 
    T *tmp = m_freeItems.top();
    m_freeItems.pop();
    return tmp;
  }

  void PutFreeItem( T *free)
  {
    m_freeItems.push( free);
  }
};

#endif