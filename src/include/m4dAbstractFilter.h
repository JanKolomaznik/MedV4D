#ifndef __M4D_ABSTRACT_FILTER_H_
#define __M4D_ABSTRACT_FILTER_H_

#include <vtkGenericDataSetAlgorithm.h>

namespace vtkIntegration
{
	
class m4dAbstractFilter: public vtkGenericDataSetAlgorithm
{
public:

protected:

private:

public: /*inherited from VTK*/
	vtkTypeRevisionMacro(vtkGenericDataSetAlgorithm,vtkAlgorithm);
	void PrintSelf(ostream& os, vtkIndent indent);

	// Description:
	// Get the output data object for a port on this algorithm.
	vtkGenericDataSet* GetOutput();
	vtkGenericDataSet* GetOutput(int);
	virtual void SetOutput(vtkDataObject* d);

	// Description:
	// see vtkAlgorithm for details
	virtual int ProcessRequest(vtkInformation*,
			     vtkInformationVector**,
			     vtkInformationVector*);

	// this method is not recommended for use, but lots of old style filters
	// use it
	vtkDataObject* GetInput();
	vtkDataObject *GetInput(int port);
	vtkGenericDataSet *GetGenericDataSetInput(int port);

	// Description:
	// Set an input of this algorithm. You should not override these
	// methods because they are not the only way to connect a pipeline.
	// Note that these methods support old-style pipeline connections.
	// When writing new code you should use the more general
	// vtkAlgorithm::SetInputConnection().  These methods transform the
	// input index to the input port index, not an index of a connection
	// within a single port.
	void SetInput(vtkDataObject *);
	void SetInput(int, vtkDataObject*);

	// Description:
	// Add an input of this algorithm.  Note that these methods support
	// old-style pipeline connections.  When writing new code you should
	// use the more general vtkAlgorithm::AddInputConnection().  See
	// SetInput() for details.
	void AddInput(vtkDataObject *);
	void AddInput(int, vtkDataObject*);

protected:
	m4dAbstractFilter();
	~m4dAbstractFilter();

	// convenience method
	virtual int RequestInformation(vtkInformation* request,
				 vtkInformationVector** inputVector,
				 vtkInformationVector* outputVector);

	// Description:
	// This is called by the superclass.
	// This is the method you should override.
	// See ProcessRequest for details about arguments and return value.
	virtual int RequestData(vtkInformation* request,
			  vtkInformationVector** inputVector,
			  vtkInformationVector* outputVector);

	// Description:
	// This is called by the superclass.
	// This is the method you should override.
	// See ProcessRequest for details about arguments and return value.
	virtual int RequestDataObject(vtkInformation* request,
				vtkInformationVector** inputVector,
				vtkInformationVector* outputVector)=0;

	// Description:
	// This is called by the superclass.
	// This is the method you should override.
	// See ProcessRequest for details about arguments and return value.
	virtual int RequestUpdateExtent(vtkInformation*,
				  vtkInformationVector**,
				  vtkInformationVector*);

	// Description:
	// This method is the old style execute method
	virtual void ExecuteData(vtkDataObject *output);
	virtual void Execute();

	// see algorithm for more info
	virtual int FillOutputPortInformation(int port, vtkInformation* info);
	virtual int FillInputPortInformation(int port, vtkInformation* info);

private:
	m4dAbstractFilter(const m4dAbstractFilte&);  // Not implemented.
	void operator=(const m4dAbstractFilter&);  // Not implemented.
};

} /*namespace vtkIntegration*/
#endif /*__M4D_ABSTRACT_FILTER_H_*/
