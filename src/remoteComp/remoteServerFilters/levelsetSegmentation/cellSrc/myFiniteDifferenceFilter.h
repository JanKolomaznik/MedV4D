#ifndef MYFINITEDIFFERENCEFILTER_
#define MYFINITEDIFFERENCEFILTER_

#include "itkInPlaceImageFilter.h"
#include "commonConsts.h"

namespace M4D
{
namespace Cell
{

template <class TInputImage, class TOutputImage> class MyFiniteDifferenceImageFilter :
	public itk::InPlaceImageFilter<TInputImage, TOutputImage>, public Consts
{
public:
	/** Standard class typedefs. */
	typedef MyFiniteDifferenceImageFilter Self;
	typedef itk::InPlaceImageFilter<TInputImage, TOutputImage> Superclass;
	typedef itk::SmartPointer<Self> Pointer;

	/** Run-time type information (and related methods) */
	itkTypeMacro(MyFiniteDifferenceImageFilter, InPlaceImageFilter);

	/** Input and output image types. */
	typedef TInputImage InputImageType;
	typedef TOutputImage OutputImageType;

	/** Dimensionality of input and output data is assumed to be the same. */
	itkStaticConstMacro(ImageDimension, unsigned int,
			OutputImageType::ImageDimension);

	/** The pixel type of the output image will be used in computations. */
	typedef typename TOutputImage::PixelType OutputPixelType;
	typedef typename TInputImage::PixelType InputPixelType;
	typedef OutputPixelType PixelType;
	typedef OutputPixelType ValueType;

	/** Extract value type in case the pixel is of vector type */
	typedef typename itk::NumericTraits< OutputPixelType>::ValueType
			OutputPixelValueType;
	typedef typename itk::NumericTraits< InputPixelType>::ValueType
			InputPixelValueType;

	typedef enum
	{	UNINITIALIZED = 0, INITIALIZED = 1} FilterStateType;

	/** Get the number of elapsed iterations of the filter. */
	itkGetConstReferenceMacro(ElapsedIterations, unsigned int);

	/** Set/Get the number of iterations that the filter will run. */
	itkSetMacro(NumberOfIterations, unsigned int);
	itkGetConstReferenceMacro(NumberOfIterations, unsigned int);

	/** Use the image spacing information in calculations. Use this option if you
	 *  want derivatives in physical space. Default is UseImageSpacingOff. */
	itkSetMacro(UseImageSpacing, bool);
	itkBooleanMacro(UseImageSpacing);
	itkGetConstReferenceMacro(UseImageSpacing, bool);

	/** Set/Get the maximum error allowed in the solution.  This may not be
	 defined for all solvers and its meaning may change with the application. */
	itkSetMacro(MaximumRMSError, double);
	itkGetConstReferenceMacro(MaximumRMSError, double);

	/** Set/Get the root mean squared change of the previous iteration. May not
	 be used by all solvers. */
	itkSetMacro(RMSChange, double);
	itkGetConstReferenceMacro(RMSChange, double);

	/** Set the state of the filter to INITIALIZED */
	void SetStateToInitialized()
	{
		this->SetState(INITIALIZED);
	}

	/** Set the state of the filter to UNINITIALIZED */
	void SetStateToUninitialized()
	{
		this->SetState(UNINITIALIZED);
	}

	/** Set/Get the state of the filter. */
#if !defined(CABLE_CONFIGURATION)
	itkSetMacro(State, FilterStateType);
	itkGetConstReferenceMacro(State, FilterStateType);
#endif

	/** Require the filter to be manually reinitialized (by calling
	 SetStateToUninitialized() */
	itkSetMacro(ManualReinitialization, bool);
	itkGetConstReferenceMacro(ManualReinitialization, bool);
	itkBooleanMacro(ManualReinitialization);

#ifdef ITK_USE_STRICT_CONCEPT_CHECKING
	/** Begin concept checking */
	itkConceptMacro(OutputPixelIsFloatingPointCheck,
			(Concept::IsFloatingPoint<OutputPixelValueType>));
	/** End concept checking */
#endif

protected:
	MyFiniteDifferenceImageFilter()
	{
		m_UseImageSpacing = false;
		m_ElapsedIterations = 0;
		m_NumberOfIterations = itk::NumericTraits<unsigned int>::max();
		m_MaximumRMSError = 0.0;
		m_RMSChange = 0.0;
		m_State = UNINITIALIZED;
		m_ManualReinitialization = false;
		this->InPlaceOn();
	}
	~MyFiniteDifferenceImageFilter()
	{
	}

	/** This method is defined by a subclass to apply changes to the output
	 * from an update buffer and a time step value "dt".
	 * \param dt Time step value. */
	virtual void ApplyUpdate(TimeStepType dt) = 0;

	/** This method is defined by a subclass to populate an update buffer
	 * with changes for the pixels in the output.  It returns a time
	 * step value to be used for the update.
	 * \returns A time step to use in updating the output with the changes
	 * calculated from this method. */
	virtual TimeStepType CalculateChange() = 0;

	/** This is the default, high-level algorithm for calculating finite
	 * difference solutions.  It calls virtual methods in its subclasses
	 * to implement the major steps of the algorithm. */
	virtual void GenerateData();

	/** MyFiniteDifferenceImageFilter needs a larger input requested region than
	 * the output requested region.  As such, we need to provide
	 * an implementation for GenerateInputRequestedRegion() in order to inform
	 * the pipeline execution model.
	 *
	 * \par
	 * The filter will ask for a padded region to perform its neighborhood
	 * calculations.  If no such region is available, the boundaries will be
	 * handled as described in the FiniteDifferenceFunction defined by the
	 * subclass.
	 * \sa ProcessObject::GenerateInputRequestedRegion() */
	virtual void GenerateInputRequestedRegion();

	virtual void Initialize()=0;

	/** This method returns true when the current iterative solution of the
	 * equation has met the criteria to stop solving.  Defined by a subclass. */
	virtual bool Halt();

	/** This method is similar to Halt(), and its default implementation in this
	 * class is simply to call Halt(). However, this method takes as a parameter
	 * a void pointer to the MultiThreader::ThreadInfoStruct structure. If you
	 * override this method instead of overriding Halt, you will be able to get
	 * the current thread ID and handle the Halt method accordingly. This is useful
	 * if you are doing a lot of processing in Halt that you don't want parallelized.
	 * Notice that ThreadedHalt is only called by the multithreaded filters, so you
	 * still should implement Halt, just in case a non-threaded filter is used.
	 */
	virtual bool ThreadedHalt(void *itkNotUsed(threadInfo))
	{
		return this->Halt();
	}

	virtual void CopyInputToOutput(void) = 0;
	/** Virtual method for resolving a single time step from a set of time steps
	 * returned from processing threads.
	 * \return Time step (dt) for the iteration update based on a list
	 * of time steps generated from the threaded calculated change method (one
	 * for each region processed).
	 *
	 * \param timeStepList The set of time changes compiled from all the threaded calls
	 * to ThreadedGenerateData.
	 * \param valid The set of flags indicating which of "list" elements are
	 *  valid
	 * \param size The size of "list" and "valid"
	 *
	 * The default is to return the minimum value in the list. */
	virtual TimeStepType ResolveTimeStep(const TimeStepType* timeStepList,
			const bool* valid, int size);

	/** Set the number of elapsed iterations of the filter. */
	itkSetMacro(ElapsedIterations, unsigned int);

	/** This method is called after the solution has been generated to allow
	 * subclasses to apply some further processing to the output.*/
	virtual void PostProcessOutput()
	{
	}

	/** The maximum number of iterations this filter will run */
	unsigned int m_NumberOfIterations;

	/** A counter for keeping track of the number of elapsed 
	 iterations during filtering. */
	unsigned int m_ElapsedIterations;

	/** Indicates whether the filter automatically resets to UNINITIALIZED state
	 after completing, or whether filter must be manually reset */
	bool m_ManualReinitialization;

	double m_RMSChange;
	double m_MaximumRMSError;

private:
	MyFiniteDifferenceImageFilter(const Self&); //purposely not implemented
	void operator=(const Self&); //purposely not implemented


	/** Control whether derivatives use spacing of the input image in
	 its calculation. */
	bool m_UseImageSpacing;

	/** State that the filter is in, i.e. UNINITIALIZED or INITIALIZED */
	FilterStateType m_State;
};

//include implementation
#include "src/myFiniteDifferenceFilter.tcc"

}
}

#endif /*MYFINITEDIFFERENCEFILTER_*/
