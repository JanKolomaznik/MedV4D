/**
 * @author Attila Ulman
 * @file PerfusionAnalysisFilter.h
 * @{ 
 **/

#ifndef PERFUSION_ANALYSIS_FILTER_H
#define PERFUSION_ANALYSIS_FILTER_H

#include "Imaging/AImageFilterWholeAtOnce.h"


namespace M4D {
namespace Imaging {

#define KERNEL_SIZE             7

#define MAX_PERCENTAGE_PE       0.13
#define MAX_PERCENTAGE_TTP      0.7
#define MAX_PERCENTAGE_AT       0.8
#define MAX_PERCENTAGE_CBV      0.2
#define MAX_PERCENTAGE_MTT      0.6
#define MAX_PERCENTAGE_CBF      0.2
#define MAX_PERCENTAGE_US       0.2
#define MAX_PERCENTAGE_DS       0.2

#define SLOPE_VIS_FACTOR        10

#define MEDIAN_FILTER_RADIUS    5

/// Enumeration for the visualization type.
enum VisualizationType {
  /// maximum intensity projection
	VT_MAX,
  /// subtraction images
	VT_SUBTR,
  /// parameter maps
	VT_PARAM
};

/// Enumeration for the parameter map type.
enum ParameterType {
  /// peak enhancement
  PT_PE,
	/// time to peak
  PT_TTP,
	/// arrival time
  PT_AT,
  /// cerebal blood volume
  PT_CBV,
  /// mean transit time
  PT_MTT,
  /// cerebal blood flow
  PT_CBF,
  /// upslope - wash-in steepness
  PT_US,
  /// downslope - wash-out steepness
  PT_DS
};

typedef vector< ElementType > IntensitiesType;
typedef vector< int32 > SumType;
typedef ImageType::Iterator Iterator;

template< typename ElementType >
class VisOperator;

template< typename ImageType >
class PerfusionAnalysisFilter;

/**
 * Filter implementing multiscan, times series analysis, partly visualization. 
 */
template< typename ImageType >
class PerfusionAnalysisFilter
	: public AMultiImageFilter< 1, 3 >
{
  public:	

    typedef AMultiImageFilter< 1, 3 >                     PredecessorType;
    typedef typename ImageTraits< ImageType >::InputPort  ImageInPort;
	  typedef typename ImageTraits< ImageType >::OutputPort ImageOutPort;

    /**
     * Properties structure.
     */
	  struct Properties: public PredecessorType::Properties
	  {
      /**
       * Properties constructor - fills up the Properties with default values.
       */
		  Properties ()
        : examinedSliceNum( EXEMINED_SLICE_NUM ), medianFilterRadius( 0 ), visualizationType( VT_MAX ),
          intensitiesSize( 0 ), maxCASliceIndex( 0 ), subtrIndexLow( -1 ), subtrIndexHigh( -1 ),
          parameterType( PT_PE ), minParameterValue( 0 ), maxParameterValue( 0 ),
          minValuePercentage( 0 ), maxValuePercentage( MAX_PERCENTAGE_PE ), backgroundNeeded( false )
      {}

      /// Number of examined slices (number of time series).
		  uint32 examinedSliceNum;
      /// Radius of the median filter kernel.
      uint8 medianFilterRadius;
      /// Type of the visualization.
      VisualizationType visualizationType;
      /// Number of slices in 1 time series (size of the intensity vector for point of the image).
      uint16 intensitiesSize;
      /// Index of the slice with maximum sum of its intensities ("optimal" time estimation for subtr. images).
      uint16 maxCASliceIndex;
      /// Indices of slices in time series for purposes of subtr. images.
      int16 subtrIndexLow, subtrIndexHigh;
      /// Type of the parameter map.
      ParameterType parameterType;
      /// Min. and max. values occuring in the parameter map (for color ramp calibration).
      uint16 minParameterValue, maxParameterValue;
      /// Range of the visualized parameter map.
      float32 minValuePercentage, maxValuePercentage;
      /// Flag indicating whether the generation of image also on background is needed (in case of switched on see-through).
      bool backgroundNeeded;
	  };

    /**
     * Analysis filter constructor.
     *
     * @param prop pointer to the properties structure
     */
	  PerfusionAnalysisFilter ( Properties *prop );

    /**
     * Analysis filter constructor.
     */
	  PerfusionAnalysisFilter ();

    /**
     * Analysis filter destructor.
     */
    ~PerfusionAnalysisFilter ();

    /**
     * Method visualizating the result - can be called also after pipeline execution.
     */
    void Visualize ();

    /**
     * Getter for smoothed TIC of the given point.
     *
     * @param x x coordinate of the point
     * @param y y coordinate of the point
     * @param z z coordinate of the point
     * @return the requested TIC
     */
    IntensitiesType &GetSmoothedCurve ( int x, int y, int z );

	  GET_SET_PROPERTY_METHOD_MACRO(uint32, ExaminedSliceNum, examinedSliceNum);
    GET_SET_PROPERTY_METHOD_MACRO(uint8, MedianFilterRadius, medianFilterRadius);
    GET_SET_PROPERTY_METHOD_MACRO(VisualizationType, VisualizationType, visualizationType);
    GET_SET_PROPERTY_METHOD_MACRO(uint16, IntensitiesSize, intensitiesSize);
    GET_SET_PROPERTY_METHOD_MACRO(uint16, MaxCASliceIndex, maxCASliceIndex);
    GET_SET_PROPERTY_METHOD_MACRO(int16, SubtrIndexLow, subtrIndexLow);
    GET_SET_PROPERTY_METHOD_MACRO(int16, SubtrIndexHigh, subtrIndexHigh);
    GET_SET_PROPERTY_METHOD_MACRO(ParameterType, ParameterType, parameterType);
    GET_SET_PROPERTY_METHOD_MACRO(uint16, MinParameterValue, minParameterValue);
    GET_SET_PROPERTY_METHOD_MACRO(uint16, MaxParameterValue, maxParameterValue);
    GET_SET_PROPERTY_METHOD_MACRO(float32, MinValuePercentage, minValuePercentage);
    GET_SET_PROPERTY_METHOD_MACRO(float32, MaxValuePercentage, maxValuePercentage);
    GET_SET_PROPERTY_METHOD_MACRO(bool, BackgroundNeeded, backgroundNeeded);
  
  protected:

    /**
     * Method for managing visualization functionality - called from Visualize.
     *
     * @param visOperator pointer to the visualization operator by the help of which is the method filling the output
     * @param outIdx index of the output port, where to put the result (to which layer)
     */
    void VisualizeHelper ( VisOperator< ElementType > *visOperator, uint8 outIdx );

    /**
     * Method realizing median filter.
     *
     * @param radius radius of the median filter kernel
     * @param outIdx index of the output port, where to put the result (to which layer)
     */
    void MedianFilter ( uint8 radius, uint8 outIdx );

    /**
     * This method is executed by the pipeline's filter thread.
     *
	   * @param utype update type
	   * @return true if the filter finished successfully, false otherwise
     */
	  bool ExecutionThreadMethod ( APipeFilter::UPDATE_TYPE utype );

    /**
     * It prepares output datasets according to inputs and ﬁlter properties.
     */
    void PrepareOutputDatasets ();

    /**
     * This method do all preparations before actual computation.
     * When overriding in successors predecessor implementation must be called first.
     *
	   * @param utype desired update method
     */
    void BeforeComputation ( APipeFilter::UPDATE_TYPE &utype );

    /**
     * Managing the dataset locking.
     *
	   * @param utype update type
     */
	  void MarkChanges ( APipeFilter::UPDATE_TYPE utype );

    /**
     * Method called in execution methods after computation.
     * When overriding in successors, predecessor implementation must be called as last.
     *
	   * @param successful information, whether computation proceeded without problems
     */
    void AfterComputation ( bool successful );

    /// Reader/writer BBox for locking purposes.
    ReaderBBoxInterface::Ptr readerBBox;
	  WriterBBoxInterface *writerBBox[ 3 ];

  private:

    /**
     * Actual computation method called by the ExecutionThreadMethod.
     * It's preparing TICs and calling Visualize.
     *
     * @return true if finished successfully, false otherwise
     */
    bool Process ();

	  GET_PROPERTIES_DEFINITION_MACRO;

    /// Matrices of vectors (TIC) - for every examined slice one - original and smoothed versions.
    IntensitiesType *originalIntensities, *smoothedIntensities;

    /// Dimensions of the input image.
    uint32 inWidth, inHeight;
};


/**
 * Abstract base class for all kinds of visualization operators.
 */
template< typename ElementType >
class VisOperator
{
  public:

    /**
     * Pure virtual method for computing the visualizated result according to input coordinates (index).
     * Need to be implemented in subclasses, according to the method that the subclass wishes to realize.
     *
     * @param idx index to matrices of TICs
     * @param outIt reference to the output iterator (the result is stored there) 
     */
	  virtual void Result ( uint32 idx, Iterator &outIt ) = 0;

    /// Matrices of vectors of intensities (TICs).
    IntensitiesType *intensities;
};


/**
 * As a subclass of VisOperator, this class realizes the Maximum Intensity projection type of visualization..
 */
template< typename ElementType >
class MaxVisOperator: public VisOperator< ElementType >
{
  public:

    /**
     * MaxVisOperator constructor.
     *
     * @param domain of the operator - pointer to matrices of TICs
     */
    MaxVisOperator ( IntensitiesType *domain )
    {
      intensities = domain;  
    }

    /**
     * Method for computing the visualizated result according to input coordinates (index).
     * Implementation of the base class’ Result method.
     *
     * @param idx index to matrices of TICs
     * @param outIt reference to the output iterator (the result is stored there) 
     */
    void Result ( uint32 idx, Iterator &outIt )
	  { 
      *outIt = *max_element( intensities[idx].begin(), intensities[idx].end() );
    }
};


/**
 * Subclass of VisOperator that realizes the Subtraction Images type of visualization.
 */
template< typename ElementType >
class SubtrVisOperator: public VisOperator< ElementType >
{
  public:
  
    /**
     * SubtrVisOperator constructor.
     *
     * @param domain of the operator - pointer to matrices of TICs
     * @param indexLow index of the slice which will be subtracted from another (indexHigh)
     * @param indexHigh index of the slice from which will be subtracted the another (indexLow)
     * @param maxCAIndex index of the slice with maximum sum of its intensities 
     */
    SubtrVisOperator ( IntensitiesType *domain, int16 indexLow, int16 indexHigh, uint16 maxCAIndex )
    {
      intensities = domain;

      idx1 = indexLow;
      idx2 = indexHigh;

      // no index set -> default behavior: substract first from the middle one in the sequence
      if ( idx1 == -1 ) {
        idx1 = 0;
      }
      if ( idx2 == -1 ) {
        idx2 = maxCAIndex;
      }
    }

    /**
     * Method for computing the visualizated result according to input coordinates (index).
     * Implementation of the base class’ Result method.
     *
     * @param idx index to matrices of TICs
     * @param outIt reference to the output iterator (the result is stored there) 
     */
    void Result ( uint32 idx, Iterator &outIt )
	  { 
      *outIt = intensities[idx][idx2] - intensities[idx][idx1];
    }

    /// Indices of slices in time series to be subtracted from each other.
    int16 idx1, idx2;
};


/**
 * Subclass of VisOperator that realizes the Parameter Maps type of visualization.
 */
template< typename ElementType >
class ParamVisOperator: public VisOperator< ElementType >
{
  public:

    /**
     * ParamVisOperator constructor.
     *
     * @param domain of the operator - pointer to matrices of TICs
     * @param type type of the parameter map
     * @param minPercent percent of min. value for determining the range of the visualized parameter map
     * @param maxPercent percent of max. value for determining the range of the visualized parameter map
     */
    ParamVisOperator ( IntensitiesType *domain, ParameterType type, float32 minPercent, float32 maxPercent )
    {
      intensities = domain;  

      parameterType = type;
      
      minValue = maxValue = 0;
     
      minValuePercentage = minPercent;
      maxValuePercentage = maxPercent;
    }
  
    /**
     * Method for computing the visualizated result according to input coordinates (index).
     * Implementation of the base class’ Result method.
     *
     * @param idx index to matrices of TICs
     * @param outIt reference to the output iterator (the result is stored there) 
     */
    void Result ( uint32 idx, Iterator &outIt )
	  { 
      uint16 maxIndex = max_element( intensities[idx].begin(), intensities[idx].end() ) - intensities[idx].begin();

      uint32 integral = intensities[idx][maxIndex];
      
      // calculating the CA arrival time + the integral     
      uint16 arrivalIndex = GetArrivalIndex( intensities[idx], maxIndex, integral );
      
      // calculating the baseline 
      uint32 baseline = 0;
      for ( uint16 i = 0; i < arrivalIndex; i++ ) {
        baseline += intensities[idx][i] / arrivalIndex;
      }

      // calculating the end of the first CA passage + the integral
      uint16 endIndex = GetEndIndex( intensities[idx], maxIndex, integral );

      // for CBF calculations
      ElementType CBV = 0, MTT = 0;

      // for US, DS calculations
      uint16 avgX = 0, n = 0;
      uint32 avgY = 0;
      int32 sX = 0, sXY = 0;

      switch ( parameterType ) 
      {
        case PT_PE:
          // peak enhancement - maximum value normalized by subtracting the baseline
          *outIt = baseline ? intensities[idx][maxIndex] - baseline : 0;
          break;

        case PT_TTP:
          // index of the maximum of intensities
          *outIt = maxIndex;
          break;

        case PT_AT:
          // CA arrival time (on the smoothed TIC) 
          *outIt = arrivalIndex;
          break;

        case PT_CBV:
          // integral
          *outIt = baseline ? integral - (baseline * (endIndex - arrivalIndex + 1)) : 0;
          break;
        
        case PT_MTT:
          // time transfer
          *outIt = arrivalIndex ? (endIndex - arrivalIndex + 1) / 2 : 0;
          break;

        case PT_CBF:
          // from the central volume equation, the cerebal blood flow
          CBV = baseline ? integral - (baseline * (endIndex - arrivalIndex + 1)) : 0;
          MTT = arrivalIndex ? (endIndex - arrivalIndex + 1) / 2 : 0;
          
          *outIt = MTT ? CBV / MTT : 0;  
          break;

        case PT_US:
          // wash-in steepness
          for ( uint16 i = arrivalIndex; i <= maxIndex; i++ ) 
          {
            avgX += i;
            avgY += intensities[idx][i];
          }

          n = maxIndex - arrivalIndex + 1;
          
          avgX /= n;
          avgY /= n;

          for ( uint16 i = arrivalIndex; i <= maxIndex; i++ ) 
          {
            sX  += (i - avgX) * (i - avgX);
            sXY += (i - avgX) * (intensities[idx][i] - avgY);
          }

          *outIt = sX ? ((double)sXY / sX) * SLOPE_VIS_FACTOR : 0;
          break;

        case PT_DS:
          // wash-out steepness
          for ( uint16 i = maxIndex; i <= endIndex; i++ ) 
          {
            avgX += i;
            avgY += intensities[idx][i];
          }

          n = endIndex - maxIndex + 1;
          
          avgX /= n;
          avgY /= n;

          for ( uint16 i = maxIndex; i <= endIndex; i++ ) 
          {
            sX  += (i - avgX) * (i - avgX);
            sXY += (i - avgX) * (intensities[idx][i] - avgY);
          }

          *outIt = sX ? -((double)sXY / sX) * SLOPE_VIS_FACTOR : 0;
          break;

        default:
          *outIt = baseline ? intensities[idx][maxIndex] - baseline : 0;
      }

      ElementType out = *outIt * maxValuePercentage;
      if ( out > maxValue ) {
        maxValue = out;
      }
    }

    /**
     * Determines the CA arrival time.
     *
     * @param intensities intensities vector (TIC)
     * @param maxIndex index of maximum CA value - from this point starts the searching
     * @param integral reference to the integral, which is calculated along
     * @return index (time) of the arrival 
     */
    uint16 GetArrivalIndex ( IntensitiesType &intensities, uint16 maxIndex, uint32 &integral )
    {
      ElementType prev = intensities[maxIndex];
      uint16 arrivalIndex = 0;
      
      for ( int16 i = maxIndex - 1; i > 0; i-- )
      {
        ElementType next = intensities[i];

        if ( next >= prev )
        {
          arrivalIndex = i + 1;
          break;
        }

        integral += next;

        prev = next;   
      }  

      return arrivalIndex;
    }

    /**
     * Determines the CA end time.
     *
     * @param intensities intensities vector (TIC)
     * @param maxIndex index of maximum CA value - from this point starts the searching
     * @param integral reference to the integral, which is calculated along
     * @return index (time) of the end 
     */
    uint16 GetEndIndex ( IntensitiesType &intensities, uint16 maxIndex, uint32 &integral )
    {
      ElementType prev = intensities[maxIndex];
      uint16 endIndex = 0;
      
      for ( uint16 i = maxIndex + 1; i < intensities.size(); i++ )
      {
        ElementType next = intensities[i];

        if ( next > prev ) 
        {
          endIndex = i - 1;
          break;
        }

        integral += next;

        prev = next;   
      }

      return endIndex ? endIndex : intensities.size() - 1;
    }

    /// Type of the parameter map.
    ParameterType parameterType;
    /// Min. and max. values occuring in the parameter map.
    uint16 minValue, maxValue;
    /// Range of the parameter map.
    float32 minValuePercentage, maxValuePercentage;
};

} // namespace Imaging
} // namespace M4D


// include implementation
#include "PerfusionAnalysisFilter.tcc"

#endif // PERFUSION_ANALYSIS_FILTER_H

/** @} */
