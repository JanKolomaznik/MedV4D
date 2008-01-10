#ifndef __M4D_CELL_ITERATOR_H_
#define __M4D_CELL_ITERATOR_H_

#include <vtkGenericCellIterator.h>

namespace vtkIntegration
{

class m4dCellIterator: public vtkGenericCellIterator
{
public:

protected:

private:

public:
	// Description:
	// Standard VTK construction and type macros.
	vtkTypeRevisionMacro(vtkGenericCellIterator,vtkObject);
	void PrintSelf(ostream& os, vtkIndent indent);

	// Description:
	// Move iterator to first position if any (loop initialization).
	virtual void Begin() = 0;

	// Description:
	// Is the iterator at the end of traversal?
	virtual int IsAtEnd() = 0;

	// Description:
	// Create an empty cell. The user is responsible for deleting it.
	// \post result_exists: result!=0
	virtual vtkGenericAdaptorCell *NewCell() = 0;

	// Description:
	// Get the cell at current position. The cell should be instantiated
	// with the NewCell() method.
	// \pre not_at_end: !IsAtEnd()
	// \pre c_exists: c!=0
	// THREAD SAFE
	virtual void GetCell(vtkGenericAdaptorCell *c) = 0;

	// Description:
	// Get the cell at the current traversal position.
	// NOT THREAD SAFE
	// \pre not_at_end: !IsAtEnd()
	// \post result_exits: result!=0
	virtual vtkGenericAdaptorCell *GetCell() = 0;

	// Description:
	// Move the iterator to the next position in the list.
	// \pre not_at_end: !IsAtEnd()
	virtual void Next() = 0;

protected:
	m4dCellIterator();
	virtual ~m4dCellIterator();

private:
	m4dCellIterator(const m4dCellIterator&);  // Not implemented.
	void operator=(const m4dCellIterator&);  // Not implemented.

};


} /*namespace vtkIntegration*/
#endif /*__M4D_CELL_ITERATOR_H_*/
