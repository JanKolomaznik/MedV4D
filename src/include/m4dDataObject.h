#ifndef __M4D_DATA_OBJECT_H_
#define __M4D_DATA_OBJECT_H_

#include <vtkDataObject.h>

class m4dCompatibilityAgregation;


class m4dDataObject: public vtkDataObject
{
public:

protected:
	m4dCompatibilityAgregation	*agregation;
public: /*inherited from VTK*/
	virtual const char*	GetClassName ();
	virtual int		IsA (const char *type);
	void			PrintSelf (ostream &os, vtkIndent indent);
	virtual vtkAlgorithmOutput*	GetProducerPort ();
	unsigned long int	GetMTime ();
	virtual void		Initialize ();
	void			ReleaseData ();
	int			ShouldIReleaseData ();
	virtual void		Register (vtkObjectBase *o);
	virtual void		UnRegister (vtkObjectBase *o);
	virtual void		Update ();
	virtual void		UpdateInformation ();
	virtual void		PropagateUpdateExtent ();
	virtual void		TriggerAsynchronousUpdate ();
	virtual void		UpdateData ();
	virtual unsigned long	GetEstimatedMemorySize ();
	virtual int		GetDataObjectType ();
	unsigned long		GetUpdateTime ();
	void			SetUpdateExtentToWholeExtent ();
	unsigned long		GetPipelineMTime ();
	virtual unsigned long	GetActualMemorySize ();
	void			CopyInformation (vtkDataObject *data);
	virtual void		CopyInformationFromPipeline (vtkInformation *request);
	void			DataHasBeenGenerated ();
	virtual void		PrepareForNewData ();
	virtual int		GetExtentType ();
	virtual void		Crop ();
	virtual vtkSource*	GetSource ();
	void			SetSource (vtkSource *s);
	virtual vtkInformation*	GetInformation ();
	virtual void		SetInformation (vtkInformation *);
	virtual vtkInformation*	GetPipelineInformation ();
	virtual void		SetPipelineInformation (vtkInformation *);
	virtual int		GetDataReleased ();
	void			SetReleaseDataFlag (int);
	int			GetReleaseDataFlag ();
	virtual void		ReleaseDataFlagOn ();
	virtual void		ReleaseDataFlagOff ();
	virtual void		SetFieldData (vtkFieldData *);
	virtual vtkFieldData*	GetFieldData ();
	virtual void		SetUpdateExtent (int piece, int numPieces, int ghostLevel);
	void			SetUpdateExtent (int piece, int numPieces);
	virtual void		SetUpdateExtent (int x0, int x1, int y0, int y1, int z0, int z1);
	virtual void		SetUpdateExtent (int extent[6]);
	virtual int *		GetUpdateExtent ();
	virtual void		GetUpdateExtent (int &x0, int &x1, int &y0, int &y1, int &z0, int &z1);
	virtual void		GetUpdateExtent (int extent[6]);
	virtual void		CopyTypeSpecificInformation (vtkDataObject *data);
	void			SetUpdatePiece (int piece);
	void			SetUpdateNumberOfPieces (int num);
	virtual int		GetUpdatePiece ();
	virtual int		GetUpdateNumberOfPieces ();
	void			SetUpdateGhostLevel (int level);
	virtual int		GetUpdateGhostLevel ();
	virtual void		SetRequestExactExtent (int flag);
	virtual int		GetRequestExactExtent ();
	virtual void		RequestExactExtentOn ();
	virtual void		RequestExactExtentOff ();
	virtual void		SetWholeExtent (int x0, int x1, int y0, int y1, int z0, int z1);
	virtual void		SetWholeExtent (int extent[6]);
	virtual int *		GetWholeExtent ();
	virtual void		GetWholeExtent (int &x0, int &x1, int &y0, int &y1, int &z0, int &z1);
	virtual void		GetWholeExtent (int extent[6]);
	virtual void		SetWholeBoundingBox (double x0, double x1, double y0, double y1, double z0, double z1);
	virtual void		SetWholeBoundingBox (double bb[6]);
	virtual double*		GetWholeBoundingBox ();
	virtual void		GetWholeBoundingBox (double &x0, double &x1, double &y0, double &y1, double &z0, double &z1);
	virtual void		GetWholeBoundingBox (double extent[6]);
	virtual void		SetMaximumNumberOfPieces (int);
	virtual int		GetMaximumNumberOfPieces ();
	virtual void		CopyInformationToPipeline (vtkInformation *request, vtkInformation *input);
	virtual void		ShallowCopy (vtkDataObject *src);
	virtual void		DeepCopy (vtkDataObject *src);
	void			SetExtentTranslator (vtkExtentTranslator *translator);
	vtkExtentTranslator*	GetExtentTranslator ();
};

#endif /*__M4D_DATA_OBJECT_H_*/

