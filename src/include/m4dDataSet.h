#ifndef __M4D_DATA_SET_H_
#define __M4D_DATA_SET_H_

#include <vtkGenericDataSet.h>

#include "m4dCellIterator.h"

namespace vtkIntegration
{
	
class m4dCompatibilityAgregation;


template < typename PixelType >
class m4dDataSet: public vtkGenericDataSet
{
public:

protected:
	m4dCompatibilityAgregation	*agregation;
private:

public: /*inherited from VTK*/
	void PrintSelf(ostream& os, vtkIndent indent);

	// Description:
	// Return the number of points composing the dataset. See NewPointIterator()
	// for more details.
	// \post positive_result: result>=0
	virtual vtkIdType GetNumberOfPoints() = 0;

	// Description:
	// Return the number of cells that explicitly define the dataset. See 
	// NewCellIterator() for more details.
	// \pre valid_dim_range: (dim>=-1) && (dim<=3)
	// \post positive_result: result>=0
	virtual vtkIdType GetNumberOfCells(int dim=-1) = 0;

	// Description:
	// Return -1 if the dataset is explicitly defined by cells of varying
	// dimensions or if there are no cells. If the dataset is explicitly
	// defined by cells of a unique dimension, return this dimension.
	// \post valid_range: (result>=-1) && (result<=3)
	virtual int GetCellDimension() = 0;

	// Description:
	// Get a list of types of cells in a dataset. The list consists of an array
	// of types (not necessarily in any order), with a single entry per type.
	// For example a dataset 5 triangles, 3 lines, and 100 hexahedra would
	// result a list of three entries, corresponding to the types VTK_TRIANGLE,
	// VTK_LINE, and VTK_HEXAHEDRON.
	// THIS METHOD IS THREAD SAFE IF FIRST CALLED FROM A SINGLE THREAD AND
	// THE DATASET IS NOT MODIFIED
	// \pre types_exist: types!=0
	virtual void GetCellTypes(vtkCellTypes *types);

	// Description:
	// Return an iterator to traverse cells of dimension `dim' (or all
	// dimensions if -1) that explicitly define the dataset. For instance, it
	// will return only tetrahedra if the mesh is defined by tetrahedra. If the
	// mesh is composed of two parts, one with tetrahedra and another part with
	// triangles, it will return both, but will not return the boundary edges
	// and vertices of these cells. The user is responsible for deleting the
	// iterator.
	// \pre valid_dim_range: (dim>=-1) && (dim<=3)
	// \post result_exists: result!=0
	virtual vtkGenericCellIterator *NewCellIterator(int dim=-1) = 0;

	// Description:
	// Return an iterator to traverse cell boundaries of dimension `dim' (or
	// all dimensions if -1) of the dataset.  If `exteriorOnly' is true, only
	// the exterior cell boundaries of the dataset will be returned, otherwise
	// it will return exterior and interior cell boundaries. The user is
	// responsible for deleting the iterator.
	// \pre valid_dim_range: (dim>=-1) && (dim<=2)
	// \post result_exists: result!=0
	virtual vtkGenericCellIterator *NewBoundaryIterator(int dim=-1,
						      int exteriorOnly=0) = 0;

	// Description:
	// Return an iterator to traverse the points composing the dataset; they
	// can be points that define a cell or isolated. The user is responsible
	// for deleting the iterator.
	// \post result_exists: result!=0
	virtual vtkGenericPointIterator *NewPointIterator()=0;

	// Description:
	// Locate the closest cell to position `x' (global coordinates) with
	// respect to a tolerance squared `tol2' and an initial guess `cell' (if
	// valid). The result consists in the `cell', the `subId' of the sub-cell
	// (0 if primary cell), the parametric coordinates `pcoord' of the
	// position. It returns whether the position is inside the cell or
	// not (boolean). Tolerance is used to control how close the point is to be
	// considered "in" the cell.
	// THIS METHOD IS NOT THREAD SAFE.
	// \pre not_empty: GetNumberOfCells()>0
	// \pre cell_exists: cell!=0
	// \pre positive_tolerance: tol2>0
	virtual int FindCell(double x[3],
		       vtkGenericCellIterator* &cell,
		       double tol2,
		       int &subId,
		       double pcoords[3]) = 0;

	// Description:
	// Locate the closest point `p' to position `x' (global coordinates).
	// \pre not_empty: GetNumberOfPoints()>0
	// \pre p_exists: p!=0
	virtual void FindPoint(double x[3],
			 vtkGenericPointIterator *p)=0;

	// Description:
	// Datasets are composite objects and need to check each part for their
	// modified time.
	virtual unsigned long int GetMTime();

	// Description:
	// Compute the geometry bounding box.
	virtual void ComputeBounds()=0;

	// Description:
	// Return a pointer to the geometry bounding box in the form
	// (xmin,xmax, ymin,ymax, zmin,zmax).
	// The return value is VOLATILE.
	// \post result_exists: result!=0
	virtual double *GetBounds();

	// Description:
	// Return the geometry bounding box in global coordinates in
	// the form (xmin,xmax, ymin,ymax, zmin,zmax) in the `bounds' array.
	virtual void GetBounds(double bounds[6]);

	// Description:
	// Get the center of the bounding box in global coordinates.
	// The return value is VOLATILE.
	// \post result_exists: result!=0
	virtual double *GetCenter();

	// Description:
	// Get the center of the bounding box in global coordinates.
	virtual void GetCenter(double center[3]);

	// Description:
	// Return the length of the diagonal of the bounding box.
	// \post positive_result: result>=0
	virtual double GetLength();

	// Description:
	// Get the collection of attributes associated with this dataset.
	vtkGetObjectMacro(Attributes, vtkGenericAttributeCollection);

	// Description:
	// Set/Get a cell tessellator if cells must be tessellated during
	// processing.
	// \pre tessellator_exists: tessellator!=0
	virtual void SetTessellator(vtkGenericCellTessellator *tessellator);
	vtkGetObjectMacro(Tessellator,vtkGenericCellTessellator);

	// Description:
	// Actual size of the data in kilobytes; only valid after the pipeline has
	// updated. It is guaranteed to be greater than or equal to the memory
	// required to represent the data.
	virtual unsigned long GetActualMemorySize();

	// Description:
	// Return the type of data object.
	int GetDataObjectType();

	// Description:
	// Estimated size needed after tessellation (or special operation)
	virtual vtkIdType GetEstimatedSize() = 0;

	//BTX
	// Description:
	// Retrieve an instance of this class from an information object.
	static vtkGenericDataSet* GetData(vtkInformation* info);
	static vtkGenericDataSet* GetData(vtkInformationVector* v, int i=0);
	//ETX

protected:
	// Description:
	// Constructor with uninitialized bounds (1,-1, 1,-1, 1,-1),
	// empty attribute collection and default tessellator.
	m4dDataSet();

	virtual ~m4dDataSet();

	vtkTimeStamp ComputeTime; // Time at which bounds, center, etc. computed

private:
	m4dDataSet(const m4dDataSet&);  // Not implemented.
	void operator=(const m4dDataSet&);    // Not implemented.
};

} /*namespace vtkIntegration*/

//Include template implementation.
#include "m4dDataSet.tcc"


#endif /*__M4D_DATA_SET_H_*/

