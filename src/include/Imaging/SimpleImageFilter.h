#ifndef _ABSTRACT_IMAGE_FILTER_H
#define _ABSTRACT_IMAGE_FILTER_H

#include "ExceptionBase.h"

#include "Imaging/ImageDataTemplate.h"
#include "Imaging/ImageFactory.h"

namespace M4D
{

namespace Imaging
{


/**
 * Ancestor of filters with single input and single output. 
 * Working on two dimensional data.
 **/
template< typename InputElementType, typename OutputElementType >
class Simple2DImageFilter: public AbstractFilter
{
public:

protected:

private:

};

/**
 * Ancestor of filters with single input and single output. 
 * Working on three dimensional data.
 **/
template< typename InputElementType, typename OutputElementType >
class Simple3DImageFilter: public AbstractFilter
{
public:

protected:

private:

};

/**
 * Ancestor of filters with single input and single output. 
 * Working on three dimensional data, but computation in different
 * slices is independent.
 **/
template< typename InputElementType, typename OutputElementType >
class SimpleSlicedImageFilter: public Simple3DImageFilter< InputElementType, OutputElementType >
{
public:

protected:

private:

};

} /*namespace Imaging*/
} /*namespace M4D*/


#endif /*_ABSTRACT_IMAGE_FILTER_H*/
