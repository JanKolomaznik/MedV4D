#ifndef _IMAGE_DATA_TEMPLATE_H
#error File ImageDataTemplate.tcc cannot be included directly!
#elif

namespace Images
{

template < typename ElementType >
ImageDataTemplate< ElementType >::~ImageDataTemplate()
{
	if ( _data ) {
		delete[] _data;
	}
}

} /*namespace Images*/

#endif /*_IMAGE_DATA_TEMPLATE_H*/
