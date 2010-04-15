/**
 * @author Attila Ulman
 * @file RGBGradientSliceViewerTexturePreparer.tcc 
 * @{ 
 **/

#ifndef RGB_GRADIENT_SLICEVIEWER_TEXTURE_PREPARER_H
#error File RGBGradientSliceViewerTexturePreparer.tcc cannot be included directly!
#else


namespace M4D {
namespace Viewer {

template< typename ElementType >
RGBGradientSliceViewerTexturePreparer< ElementType >::RGBGradientSliceViewerTexturePreparer ()
{
  minValue = 0;
  maxValue = MAX_VALUE;
}


template< typename ElementType >
void RGBGradientSliceViewerTexturePreparer< ElementType >::setMinMaxValue ( uint16 min, uint16 max )
{
  minValue = min;
  maxValue = max;
}


template< typename ElementType >
bool RGBGradientSliceViewerTexturePreparer< ElementType >::prepare ( const Imaging::InputPortList& inputPorts,
                                                                     uint32 &width, uint32 &height,
                                                                     GLint brightnessRate, GLint contrastRate,
                                                                     SliceOrientation so,
                                                                     uint32 slice,
                                                                     unsigned &dimension )
{
	// get the datasets
	ElementType **channels = this->getDatasetArrays( inputPorts, 3, width, height, so, slice, dimension );

	if ( !channels[0] && !channels[1] && !channels[2] )
	{
    delete [] channels;
    return false;
	}

	// set the first three input datasets as the channels of RGB
  uint32 channelSize = width * height;
  ElementType *rgb = new ElementType[ channelSize * 3 ];

  for ( uint32 i = 0; i < height; i++ ) {
	  for ( uint32 j = 0; j < width; j++ ) {
      ColorRamp( (double)(channels[0][i * width + j] - minValue) / (maxValue - minValue), rgb, i * width + j, channelSize );
    }
  }

  DrawRamp( rgb, width, height );

  double mean[ 3 ], cont[ 3 ];

  for ( uint8 k = 0; k < 3; k++ ) 
  {
    mean[k] = 0.;
    for ( uint32 i = 0; i < height; i++ ) {
      for ( uint32 j = 0; j < width; j++ ) {
        mean[k] += (double)rgb[i * width + j + k * channelSize] / (double)(width * height);
      }
    }

    cont[k] = 0.;
    for ( uint32 i = 0; i < height; i++ ) {
      for ( uint32 j = 0; j < width; j++ ) {
        cont[k] += abs( (double)(rgb[i * width + j + k * channelSize] - mean[k]) ) / (double)(width * height);
      }
    }  
  }

  // equalize arrays
  for ( uint8 k = 0; k < 3; k++ ) {
    this->adjustArrayContrastBrightness( rgb + k * channelSize, width, height, 
                                         mean[k] * 100 + brightnessRate - BRIGHTNESS_FACTOR * (MAX_VALUE + 1), 
                                         cont[k] * 100 + contrastRate   - CONTRAST_FACTOR   * (MAX_VALUE + 1) );
  }

  // cut tool enabled & click occured -> draw see-through interface
  if ( _lastClickedPositionX >= 0 ) 
  {
    this->adjustArrayContrastBrightness( channels[1], width, height, 
                                         brightnessRate - BGR_BRIGHTNESS_FACTOR * BRIGHTNESS_FACTOR * (MAX_VALUE + 1), 
                                         contrastRate + BGR_CONTRAST_FACTOR * CONTRAST_FACTOR * (MAX_VALUE + 1) );

    DrawCut( rgb, channels[1], width, height );
  }
	
  ElementType *texture = new ElementType[ channelSize * 3 ];

	// set the RGB values one after another just like OpenGL requires
	for ( uint32 i = 0; i < height; i++ ) {
	  for ( uint32 j = 0; j < width; j++ ) {
      for ( uint8 k = 0; k < 3; k++ ) {
        texture[i * width * 3 + j * 3 + k] = rgb[i * width + j + k * channelSize];
      }
    }
  }

	// prepare texture
  glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, this->oglType(), texture );

	// free temporary allocated space
  delete [] rgb;
	delete [] texture;

  if ( channels[0] ) {
    delete [] channels[0];
  }
  if ( channels[1] ) {
    delete [] channels[1];
  }
  if ( channels[2] ) {
    delete [] channels[2];
  }

	delete [] channels;

  return true;
}


template< typename ElementType >
void RGBGradientSliceViewerTexturePreparer< ElementType >::ColorRamp ( double x, ElementType *rgb, uint32 idx, uint32 channelSize )
{
  if ( x == 0 )
  {
    rgb[idx]                = 
    rgb[idx += channelSize] =
    rgb[idx += channelSize] = 0;  
  }
  else if ( x < 0.25 ) 
  {
    rgb[idx]                = 0;
    rgb[idx += channelSize] = 4 * MAX_VALUE * x;
    rgb[idx += channelSize] = MAX_VALUE;
  } 
  else if ( x < 0.5 ) 
  {
    rgb[idx]                = 0;
    rgb[idx += channelSize] = MAX_VALUE;
    rgb[idx += channelSize] = MAX_VALUE + 4 * MAX_VALUE * (0.25 - x);
  } 
  else if ( x < 0.75 ) 
  {
    rgb[idx]                = 4 * MAX_VALUE * (x - 0.5);
    rgb[idx += channelSize] = MAX_VALUE;
    rgb[idx += channelSize] = 0;
  } 
  else 
  {
    rgb[idx]                = MAX_VALUE;
    rgb[idx += channelSize] = MAX_VALUE + 4 * MAX_VALUE * (0.75 - x);
    rgb[idx += channelSize] = 0;
  }
}


template< typename ElementType >
void RGBGradientSliceViewerTexturePreparer< ElementType >::DrawCut ( ElementType *rgb, ElementType *background, 
                                                                     uint32 width, uint32 height )
{
  uint32 channelSize = width * height;

  for ( int8 y = -SEE_THROUGH_RADIUS; y <= SEE_THROUGH_RADIUS; y++ ) {
    for ( int8 x = -SEE_THROUGH_RADIUS; x <= SEE_THROUGH_RADIUS; x++ ) {
      if ( x * x + y * y <= SEE_THROUGH_RADIUS * SEE_THROUGH_RADIUS ) 
      {
        uint32 xPos = _lastClickedPositionX + x;
        uint32 yPos = _lastClickedPositionY + y;

        if ( xPos < 0 || xPos > width || yPos < 0 || yPos > height ) {
          continue;
        }

        uint32 idx = yPos * width + xPos;

        rgb[idx] = rgb[idx + channelSize] = rgb[idx + 2 * channelSize] = background[idx];
      }
    }
  }

}


template< typename ElementType >
void RGBGradientSliceViewerTexturePreparer< ElementType >::DrawRamp ( ElementType *rgb, uint32 width, uint32 height )
{
  uint32 xPos = RAMP_POS_X * width;
  uint32 yPos = RAMP_POS_Y * height;

	for ( uint32 i = yPos; i < yPos + RAMP_HEIGHT && i < height; i++ ) {
	  for ( uint32 j = xPos; j < xPos + RAMP_WIDTH && j < width; j++ ) {
      ColorRamp( (double)(i - yPos) / RAMP_HEIGHT, rgb, i * width + j, width * height );
    }
  }

  // writing min/max values to texture (to bottom/top of the ramp)
  DrawValue( minValue, xPos, yPos - FONT_HEIGHT, rgb, width, height );
  DrawValue( maxValue, xPos, yPos + RAMP_HEIGHT, rgb, width, height );
}


template< typename ElementType >
void RGBGradientSliceViewerTexturePreparer< ElementType >::DrawValue ( uint16 value, uint32 xPos, uint32 yPos, 
                                                                       ElementType *rgb, uint32 width, uint32 height )
{
  uint32 channelSize = width * height;
  uint32 x = xPos;

  ostringstream valueStr;
  valueStr << value;

  x += (int8)(2 - valueStr.str().size()) * FONT_WIDTH / 2;

  for ( uint8 ch = 0; ch < valueStr.str().size(); ch++ ) 
  {
    uint32 charIndex = valueStr.str()[ch] << 4;
    uint8 charPos = ch * FONT_WIDTH + FONT_SPACING;

    for ( uint8 i = 0; i < FONT_HEIGHT && yPos + i < height; i++ ) 
    {
	    for ( uint8 j = 0; j < FONT_WIDTH && x + charPos + j < width; j++ ) 
      {
        uint8 bitmapValue = (fontA8x16[charIndex + i] & (1 << j)) ? MAX_VALUE : 0;
        uint32 channelIndex = (yPos + i) * width + x + charPos + FONT_WIDTH - j;

        rgb[channelIndex] = rgb[channelIndex + channelSize] = rgb[channelIndex + 2 * channelSize] = bitmapValue;
      }
    }
  }
}

} // namespace Viewer
} // namespace M4D


#endif // RGB_GRADIENT_SLICEVIEWER_TEXTURE_PREPARER_H

/** @} */
