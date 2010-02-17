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

#define TTP_MAX_PERCENTAGE      0.8

#define AT_MAX_PERCENTAGE       0.8

#define CBV_MAX_PERCENTAGE      0.2
#define CBV_FACTOR              0.1

#define MTT_MAX_PERCENTAGE      0.6

#define CBF_MAX_PERCENTAGE      0.2

/// Enumeration for the visualization type
enum VisualizationType {
  /// maximum intensity projection
	VT_MAX,
  /// subtraction images
	VT_SUBTR,
  /// parameter maps
	VT_PARAM
};

/// Enumeration for the parameter map type
enum ParameterType {
	/// time to peak
  PT_TTP,
	/// arrival time
  PT_AT,
  /// cerebal blood volume
  PT_CBV,
  /// mean transit time
  PT_MTT,
  /// cerebal blood flow
  PT_CBF
};

typedef vector< ElementType > IntensitiesType;
typedef vector< int32 > SumType;
typedef vector< ImageType::Iterator > IteratorsType;

template< typename ElementType >
class VisOperator;

template< typename ImageType >
class PerfusionAnalysisFilter;

template< typename ImageType >
class PerfusionAnalysisFilter
	: public AMultiImageFilter< 1, 3 >
{
  public:	

    typedef AMultiImageFilter< 1, 3 >                     PredecessorType;
    typedef typename ImageTraits< ImageType >::InputPort  ImageInPort;
	  typedef typename ImageTraits< ImageType >::OutputPort ImageOutPort;

	  struct Properties: public PredecessorType::Properties
	  {
		  Properties ()
        : examinedSliceNum( EXEMINED_SLICE_NUM ), visualizationType( VT_MAX ),
          intensitiesSize( 0 ), maxCASliceIndex( 0 ), subtrIndexLow( -1 ), subtrIndexHigh( -1 ),
          parameterType( PT_TTP ), minParameterValue( 0 ), maxParameterValue( 0 ),
          minValuePercentage( 0 ), maxValuePercentage( TTP_MAX_PERCENTAGE )
      {}

		  uint32 examinedSliceNum;
      VisualizationType visualizationType;
      uint16 intensitiesSize, maxCASliceIndex;
      int16 subtrIndexLow, subtrIndexHigh;
      ParameterType parameterType;
      uint16 minParameterValue, maxParameterValue;
      float32 minValuePercentage, maxValuePercentage;
	  };

	  PerfusionAnalysisFilter ( Properties * prop );
	  PerfusionAnalysisFilter ();
    ~PerfusionAnalysisFilter ();

    void Visualize ();

	  GET_SET_PROPERTY_METHOD_MACRO(uint32, ExaminedSliceNum, examinedSliceNum);
    GET_SET_PROPERTY_METHOD_MACRO(VisualizationType, VisualizationType, visualizationType);
    GET_SET_PROPERTY_METHOD_MACRO(uint16, IntensitiesSize, intensitiesSize);
    GET_SET_PROPERTY_METHOD_MACRO(uint16, MaxCASliceIndex, maxCASliceIndex);
    GET_SET_PROPERTY_METHOD_MACRO(int16, SubtrIndexLow, subtrIndexLow);
    GET_SET_PROPERTY_METHOD_MACRO(int16, SubtrIndexHigh, subtrIndexHigh);
    GET_SET_PROPERTY_METHOD_MACRO(ParameterType, ParameterType, parameterType);
    GET_SET_PROPERTY_METHOD_MACRO(uint16, MinParameterValue, minParameterValue);
    GET_SET_PROPERTY_METHOD_MACRO(uint16, MaxParameterValue, maxParameterValue);
    GET_SET_PROPERTY_METHOD_MACRO(uint8, MinValuePercentage, minValuePercentage);
    GET_SET_PROPERTY_METHOD_MACRO(float32, MaxValuePercentage, maxValuePercentage);
  
  protected:

    void VisualizeHelper ( VisOperator< ElementType > *visOperator );

	  bool ExecutionThreadMethod ( APipeFilter::UPDATE_TYPE utype );

    void PrepareOutputDatasets ();

    void BeforeComputation ( APipeFilter::UPDATE_TYPE &utype );

	  void MarkChanges ( APipeFilter::UPDATE_TYPE utype );

    void AfterComputation ( bool successful );

    ReaderBBoxInterface::Ptr readerBBox;
	  WriterBBoxInterface *writerBBox[ 3 ];

  private:

    bool Process ();

	  GET_PROPERTIES_DEFINITION_MACRO;

    IntensitiesType *originalIntensities, *smoothedIntensities;
};


template< typename ElementType >
class VisOperator
{
  public:

	  virtual void Result ( uint32 idx, IteratorsType &outIt ) = 0;

    IntensitiesType *intensities;
};


template< typename ElementType >
class MaxVisOperator: public VisOperator< ElementType >
{
  public:

    MaxVisOperator ( IntensitiesType *domain )
    {
      intensities = domain;  
    }

    void Result ( uint32 idx, IteratorsType &outIt )
	  { 
      *outIt[0] = *max_element( intensities[idx].begin(), intensities[idx].end() );
    }
};


template< typename ElementType >
class SubtrVisOperator: public VisOperator< ElementType >
{
  public:
  
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

    void Result ( uint32 idx, IteratorsType &outIt )
	  { 
      *outIt[0] = intensities[idx][idx2] - intensities[idx][idx1];
    }

    int16 idx1, idx2;
};


template< typename ElementType >
class ParamVisOperator: public VisOperator< ElementType >
{
  public:

    ParamVisOperator ( IntensitiesType *domain, ParameterType type, float32 minPercent, float32 maxPercent )
    {
      intensities = domain;  

      parameterType = type;
      
      minValue = maxValue = 0;
     
      minValuePercentage = minPercent;
      maxValuePercentage = maxPercent;
    }
  
    void Result ( uint32 idx, IteratorsType &outIt )
	  { 
      uint16 maxIndex = max_element( intensities[idx].begin(), intensities[idx].end() ) - intensities[idx].begin();

      uint32 integral = intensities[idx][maxIndex];
      
      // calculating the CA arrival time + the integral     
      uint16 arrivalIndex = GetArrivalIndex( intensities[idx], maxIndex, integral );
      
      // calculating the baseline 
      uint32 baseline = 0;
      for ( int16 i = 0; i < arrivalIndex; i++ ) {
        baseline += intensities[idx][i] / arrivalIndex;
      }

      // calculating the end of the first CA passage + the integral
      uint16 endIndex = GetEndIndex( intensities[idx], maxIndex, integral );

      ElementType CBV = 0, MTT = 0;

      switch ( parameterType ) 
      {
        case PT_TTP:
          // index of the maximum of intensities
          *outIt[0] = maxIndex;
          break;

        case PT_AT:
          // CA arrival time (on the smoothed TIC) 
          *outIt[0] = arrivalIndex;
          break;

        case PT_CBV:
          // integral
          CBV = baseline ? integral - (baseline * (endIndex - arrivalIndex + 1)) : 0;

          *outIt[0] = CBV * CBV_FACTOR;
          break;
        
        case PT_MTT:
          // time transfer
          MTT = arrivalIndex ? (endIndex - arrivalIndex + 1) / 2 : 0;

          *outIt[0] = MTT;
          break;

        case PT_CBF:
          // from the central volume equation, the cerebal blood flow
          CBV = baseline ? integral - (baseline * (endIndex - arrivalIndex + 1)) : 0;
          MTT = arrivalIndex ? (endIndex - arrivalIndex + 1) / 2 : 0;
          
          *outIt[0] = MTT ? CBV / MTT : 0;  
          break;

        default:
          *outIt[0] = maxIndex;
      }

      ElementType out = *outIt[0] * maxValuePercentage;
      if ( out > maxValue ) {
        maxValue = out;
      }
    }

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

      return endIndex ? endIndex : intensities.size();
    }

    ParameterType parameterType;
    uint16 minValue, maxValue;
    float32 minValuePercentage, maxValuePercentage;
};

} // namespace Imaging
} // namespace M4D


// include implementation
#include "PerfusionAnalysisFilter.tcc"

#endif // PERFUSION_ANALYSIS_FILTER_H

/** @} */
