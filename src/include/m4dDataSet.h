#ifndef __M4D_DATA_SET_H_
#define __M4D_DATA_SET_H_

#include <vtkDataSet.h>

class m4dCompatibilityAgregation;


class m4dDataSet: public vtkDataSet
{
public:

protected:
	m4dCompatibilityAgregation	*agregation;
private:

public: /*inherited from VTK*/
	virtual const char*	GetClassName ();
	virtual int		IsA (const char *type);
	void 			PrintSelf (ostream &os, vtkIndent indent);
	virtual void 		CopyStructure (vtkDataSet *ds)=0;
	virtual vtkIdType 	GetNumberOfPoints ()=0;
	virtual vtkIdType 	GetNumberOfCells ()=0;
	virtual double*		GetPoint (vtkIdType ptId)=0;
	virtual void		GetPoint (vtkIdType id, double x[3]);
	virtual vtkCell*	GetCell (vtkIdType cellId)=0;
	virtual void		GetCell (vtkIdType cellId, vtkGenericCell *cell)=0;
	virtual void		GetCellBounds (vtkIdType cellId, double bounds[6]);
	virtual int		GetCellType (vtkIdType cellId)=0;
	virtual void		GetCellTypes (vtkCellTypes *types);
	virtual void		GetCellPoints (vtkIdType cellId, vtkIdList *ptIds)=0;
	virtual void		GetPointCells (vtkIdType ptId, vtkIdList *cellIds)=0;
	unsigned long int 	GetMTime ();

/*PROBLEM*/	vtkCellData*		GetCellData ();
/*PROBLEM*/	vtkPointData*		GetPointData ();

	virtual void		Squeeze ();
	virtual void		ComputeBounds ();
	double*		 	GetBounds ();
	void 			GetBounds (double bounds[6]);
	double * 		GetCenter ();
	void 			GetCenter (double center[3]);
	double 			GetLength ();
	void 			Initialize ();
	virtual void 		GetScalarRange (double range[2]);
	double * 		GetScalarRange ();
	virtual int 		GetMaxCellSize ()=0;
	unsigned long 		GetActualMemorySize ();
	int 			CheckAttributes ();
	virtual void 		GenerateGhostLevelArray ();
	virtual void 		GetCellNeighbors (
					vtkIdType 	cellId, 
					vtkIdList	*ptIds, 
					vtkIdList	*cellIds
				);
	vtkIdType 		FindPoint (double x, double y, double z);
	virtual vtkIdType 	FindPoint (double x[3])=0;
	virtual vtkIdType 	FindCell (
					double 		x[3], 
					vtkCell 	*cell, 
					vtkIdType 	cellId, 
					double		tol2, 
					int		&subId, 
					double		pcoords[3], 
					double 		*weights
				)=0;
	virtual vtkIdType 	FindCell (
					double 	x[3], 
					vtkCell 	*cell, 
					vtkGenericCell	*gencell, 
					vtkIdType 	cellId, 
					double 		tol2, 
					int 		&subId, 
					double		pcoords[3], 
					double 		*weights
				)=0;
	virtual vtkCell * 	FindAndGetCell (
					double 		x[3], 
					vtkCell 	*cell, 
					vtkIdType 	cellId, 
					double 		tol2, 
					int 		&subId, 
					double 		pcoords[3], 
					double 		*weights
				);
	int 			GetDataObjectType ();
	void 			ShallowCopy (vtkDataObject *src);
	void 			DeepCopy (vtkDataObject *src);
};



#endif /*__M4D_DATA_SET_H_*/

