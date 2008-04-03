#ifndef __M4D_DATA_SET_H_
#error File m4dDataSet.tcc cannot be included directly!
#elif

namespace vtkIntegration
{

template< typename CellType >
void
m4dDataSet< CellType >::PrintSelf(ostream& os, vtkIndent indent)
{

}


template< typename CellType >
vtkIdType 
m4dDataSet< CellType >::GetNumberOfPoints()
{

}
 

template< typename CellType >
vtkIdType 
m4dDataSet< CellType >::GetNumberOfCells(int dim)
{

}
 

template< typename CellType >
int 
m4dDataSet< CellType >::GetCellDimension()
{

}
 

template< typename CellType >
void 
m4dDataSet< CellType >::GetCellTypes(vtkCellTypes *types)
{

}


template< typename CellType >
vtkGenericCellIterator *
m4dDataSet< CellType >::NewCellIterator(int dim)
{

}
 

template< typename CellType >
vtkGenericCellIterator *
m4dDataSet< CellType >::NewBoundaryIterator(int dim, int exteriorOnly)
{

}
 

template< typename CellType >
vtkGenericPointIterator *
m4dDataSet< CellType >::NewPointIterator()
{

}


template< typename CellType >
int 
m4dDataSet< CellType >::FindCell(double x[3],
       vtkGenericCellIterator* &cell,
       double tol2,
       int &subId,
       double pcoords[3])
{

}
 

template< typename CellType >
void
m4dDataSet< CellType >::FindPoint(double x[3],
	 vtkGenericPointIterator *p)
{

}


template< typename CellType >
unsigned long int 
m4dDataSet< CellType >::GetMTime()
{

}


template< typename CellType >
void 
m4dDataSet< CellType >::ComputeBounds()
{

}


template< typename CellType >
double *
m4dDataSet< CellType >::GetBounds()
{

}


template< typename CellType >
void 
m4dDataSet< CellType >::GetBounds(double bounds[6])
{

}


template< typename CellType >
double *
m4dDataSet< CellType >::GetCenter()
{

}


template< typename CellType >
void 
m4dDataSet< CellType >::GetCenter(double center[3])
{

}


template< typename CellType >
double 
m4dDataSet< CellType >::GetLength()
{

}

template< typename CellType >
void 
m4dDataSet< CellType >::SetTessellator(vtkGenericCellTessellator *tessellator)
{

}

template< typename CellType >
unsigned long 
m4dDataSet< CellType >::GetActualMemorySize()
{

}


template< typename CellType >
int 
m4dDataSet< CellType >::GetDataObjectType()
{

}


template< typename CellType >
vtkIdType 
m4dDataSet< CellType >::GetEstimatedSize()
{

}
 

} /*namespace vtkIntegration*/

#endif /*__M4D_DATA_SET_H_*/
