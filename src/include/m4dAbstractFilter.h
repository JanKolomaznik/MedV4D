#ifndef __M4D_ABSTRACT_FILTER_H_
#define __M4D_ABSTRACT_FILTER_H_

#include <vtkAlgorithm.h>

class m4dAbstractFilter: public vtkAlgorithm
{
public:

protected:

private:

public: /*inherited from VTK*/
	virtual const char*	GetClassName ()
	virtual int		IsA (const char *type);
	void			PrintSelf (ostream &os, vtkIndent indent);
	int			HasExecutive ();
	vtkExecutive*		GetExecutive ();
	virtual void		SetExecutive (vtkExecutive *executive);
	virtual int		ModifyRequest (vtkInformation *request, int when);
	vtkInformation*		GetInputPortInformation (int port);
	vtkInformation*		GetOutputPortInformation (int port);
	int			GetNumberOfInputPorts ();
	int			GetNumberOfOutputPorts ();
	void			UpdateProgress (double amount);
	vtkInformation*		GetInputArrayInformation (int idx);
	void			RemoveAllInputs ();
	vtkDataObject*		GetOutputDataObject (int port);
	virtual void		RemoveInputConnection (int port, vtkAlgorithmOutput *input);
	int			GetNumberOfInputConnections (int port);
	int			GetTotalNumberOfInputConnections ();
	vtkAlgorithmOutput*	GetInputConnection (int port, int index);
	virtual void		Update ();
	virtual void		UpdateInformation ();
	virtual void		UpdateWholeExtent ();
	void			ConvertTotalInputToPortConnection (int ind, int &port, int &conn);
	virtual int		ProcessRequest (
					vtkInformation 		*request, 
					vtkInformationVector 	**inInfo, 
					vtkInformationVector 	*outInfo
				);
	virtual int		ComputePipelineMTime (
					vtkInformation 		*request, 
					vtkInformationVector 	**inInfoVec, 
					vtkInformationVector 	*outInfoVec, 
					int 			requestFromOutputPort, 
					unsigned long 		*mtime
				);
	virtual vtkInformation *	GetInformation ();
	virtual void		SetInformation (vtkInformation *);
	virtual void		Register (vtkObjectBase *o);
	virtual void		UnRegister (vtkObjectBase *o);
	virtual void		SetAbortExecute (int);
	virtual int		GetAbortExecute ();
	virtual void		AbortExecuteOn ();
	virtual void		AbortExecuteOff ();
	virtual void		SetProgress (double);
	virtual double		GetProgress ();
	virtual void		SetProgressText (const char *);
	virtual char*		GetProgressText ();
	virtual unsigned long	GetErrorCode ();
	void			SetInputArrayToProcess (
					int 		idx, 
					int 		port, 
					int 		connection, 
					int 		fieldAssociation, 
					const char 	*name
				);
	void			SetInputArrayToProcess (
					int 		idx, 
					int 		port, 
					int 		connection, 
					int 		fieldAssociation, 
					int 		fieldAttributeType
				);
	void			SetInputArrayToProcess (int idx, vtkInformation *info);
	void			SetInputArrayToProcess (
					int 		idx, 
					int 		port, 
					int 		connection, 
					const char 	*fieldAssociation, 
					const char 	*attributeTypeorName
				);
	virtual void		SetInputConnection (int port, vtkAlgorithmOutput *input);
	virtual void		SetInputConnection (vtkAlgorithmOutput *input);
	virtual void		AddInputConnection (int port, vtkAlgorithmOutput *input);
	virtual void		AddInputConnection (vtkAlgorithmOutput *input);
	vtkAlgorithmOutput*	GetOutputPort (int index);
	vtkAlgorithmOutput*	GetOutputPort ();
	virtual void		SetReleaseDataFlag (int);
	virtual int		GetReleaseDataFlag ();
	void			ReleaseDataFlagOn ();
	void			ReleaseDataFlagOff ();
	int			UpdateExtentIsEmpty (vtkDataObject *output);
	int			UpdateExtentIsEmpty (vtkInformation *pinfo, int extentType);
};

#endif /*__M4D_ABSTRACT_FILTER_H_*/
