#ifndef __M4D_CELL_H_
#error File m4dCell.tcc cannot be included directly!
#elif
template< typename CellType >
namespace vtkIntegration
{

template< typename CellType >
void 
m4dCell< CellType >::PrintSelf(ostream& os, vtkIndent indent)
{

}

template< typename CellType >
vtkIdType 
m4dCell< CellType >::GetId()
{

}

template< typename CellType >
int 
m4dCell< CellType >::IsInDataSet()
{

}

template< typename CellType >
int 
m4dCell< CellType >::GetType()
{

}

template< typename CellType >
int
m4dCell< CellType >::GetDimension()
{

}

template< typename CellType >
int
m4dCell< CellType >::GetGeometryOrder()
{

}

template< typename CellType >
int
m4dCell< CellType >::IsGeometryLinear()
{

}

template< typename CellType >
int
m4dCell< CellType >::GetAttributeOrder(vtkGenericAttribute *a)
{

}

template< typename CellType >
int
m4dCell< CellType >::GetHighestOrderAttribute(vtkGenericAttributeCollection *ac)
{

}

template< typename CellType >
int
m4dCell< CellType >::IsAttributeLinear(vtkGenericAttribute *a)
{

}

template< typename CellType >
int
m4dCell< CellType >::IsPrimary()
{

}

template< typename CellType >
int
m4dCell< CellType >::GetNumberOfPoints()
{

}

template< typename CellType >
int
m4dCell< CellType >::GetNumberOfBoundaries(int dim=-1)
{

}

template< typename CellType >
int
m4dCell< CellType >::GetNumberOfDOFNodes()
{

}

template< typename CellType >
void 
m4dCell< CellType >::GetPointIterator(vtkGenericPointIterator *it)
{

}

template< typename CellType >
vtkGenericCellIterator *
m4dCell< CellType >::NewCellIterator()
{

}

template< typename CellType >
void 
m4dCell< CellType >::GetBoundaryIterator(vtkGenericCellIterator *boundaries, int dim=-1)
{

}

template< typename CellType >
int
m4dCell< CellType >::CountNeighbors(vtkGenericAdaptorCell *boundary)
{

}

template< typename CellType >
void 
m4dCell< CellType >::CountEdgeNeighbors( int* sharing )
{

}

template< typename CellType >
void 
m4dCell< CellType >::GetNeighbors(vtkGenericAdaptorCell *boundary,
			    vtkGenericCellIterator *neighbors)
{

}

template< typename CellType >
int
m4dCell< CellType >::FindClosestBoundary(int subId,
				  double pcoords[3],
				  vtkGenericCellIterator* &boundary)
{

}

template< typename CellType >
int
m4dCell< CellType >::EvaluatePosition(double x[3],
			       double *closestPoint, 
			       int &subId,
			       double pcoords[3], 
			       double &dist2)
{

}

template< typename CellType >
void 
m4dCell< CellType >::EvaluateLocation(int subId,
				double pcoords[3],
				double x[3])
{

}

template< typename CellType >
void 
m4dCell< CellType >::InterpolateTuple(vtkGenericAttribute *a, double pcoords[3],
				double *val)
{

}

template< typename CellType >
void 
m4dCell< CellType >::InterpolateTuple(vtkGenericAttributeCollection *c,
				double pcoords[3],
				double *val)
{

}

template< typename CellType >
void 
m4dCell< CellType >::Contour(vtkContourValues *values,
		       vtkImplicitFunction *f,
		       vtkGenericAttributeCollection *attributes,
		       vtkGenericCellTessellator *tess,
		       vtkPointLocator *locator,
		       vtkCellArray *verts,
		       vtkCellArray *lines,
		       vtkCellArray *polys,
		       vtkPointData *outPd,
		       vtkCellData *outCd,
		       vtkPointData *internalPd,
		       vtkPointData *secondaryPd,
		       vtkCellData *secondaryCd)
{

}

template< typename CellType >
void 
m4dCell< CellType >::Clip(double value, 
		    vtkImplicitFunction *f,
		    vtkGenericAttributeCollection *attributes,
		    vtkGenericCellTessellator *tess,
		    int insideOut,
		    vtkPointLocator *locator, 
		    vtkCellArray *connectivity,
		    vtkPointData *outPd,
		    vtkCellData *outCd,
		    vtkPointData *internalPd,
		    vtkPointData *secondaryPd,
		    vtkCellData *secondaryCd)
{

}

template< typename CellType >
int
m4dCell< CellType >::IntersectWithLine(double p1[3],
				double p2[3], 
				double tol,
				double &t,
				double x[3], 
				double pcoords[3],
				int &subId)
{

}

template< typename CellType >
void 
m4dCell< CellType >::Derivatives(int subId,
			   double pcoords[3],
			   vtkGenericAttribute *attribute,
			   double *derivs)
{

}

template< typename CellType >
void 
m4dCell< CellType >::GetBounds(double bounds[6])
{

}

template< typename CellType >
double *
m4dCell< CellType >::GetBounds()
{

}

template< typename CellType >
double 
m4dCell< CellType >::GetLength2()
{

}

template< typename CellType >
int
m4dCell< CellType >::GetParametricCenter(double pcoords[3])
{

}

template< typename CellType >
double 
m4dCell< CellType >::GetParametricDistance(double pcoords[3])
{

}

template< typename CellType >
double *
m4dCell< CellType >::GetParametricCoords()
{

}

template< typename CellType >
void 
m4dCell< CellType >::Tessellate(vtkGenericAttributeCollection *attributes, 
			  vtkGenericCellTessellator *tess,
			  vtkPoints *points,
			  vtkPointLocator *locator,
			  vtkCellArray* cellArray,
			  vtkPointData *internalPd,
			  vtkPointData *pd, vtkCellData* cd,
			  vtkUnsignedCharArray *types)
{

}

template< typename CellType >
int
m4dCell< CellType >::IsFaceOnBoundary(vtkIdType faceId)
{

}

template< typename CellType >
int
m4dCell< CellType >::IsOnBoundary()
{

}

template< typename CellType >
void 
m4dCell< CellType >::GetPointIds(vtkIdType *id)
{

}

template< typename CellType >
void 
m4dCell< CellType >::TriangulateFace(vtkGenericAttributeCollection *attributes,
			       vtkGenericCellTessellator *tess, int index, 
			       vtkPoints *points,
			       vtkPointLocator *locator,
			       vtkCellArray *cellArray,
			       vtkPointData *internalPd,
			       vtkPointData *pd, vtkCellData *cd )
{

}

template< typename CellType >
int *
m4dCell< CellType >::GetFaceArray(int faceId)
{

}

template< typename CellType >
int
m4dCell< CellType >::GetNumberOfVerticesOnFace(int faceId)
{

}

template< typename CellType >
int *
m4dCell< CellType >::GetEdgeArray(int edgeId)
{

}

template< typename CellType >
m4dCell< CellType >::m4dCell()
{

}

template< typename CellType >
m4dCell< CellType >::~m4dCell()
{

}

} /*namespace vtkIntegration*/
#endif /*__M4D_CELL_H_*/
