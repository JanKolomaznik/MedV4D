#ifndef PROXY_RENDERING_EXTENSION_H
#define PROXY_RENDERING_EXTENSION_H

#include "GUI/widgets/GeneralViewer.h"
#include <boost/bind.hpp>
#include <algorithm>


class ProxyRenderingExtension: public M4D::GUI::Viewer::RenderingExtension
{
public:
	typedef boost::shared_ptr< ProxyRenderingExtension > Ptr;

	unsigned
	getAvailableViewTypes()const
	{

	}

	void
	render2DAlignedSlices( int32 aSliceIdx, Vector2f aInterval, CartesianPlanes aPlane )
	{
		std::for_each( mExtensionList.begin(), mExtensionList.end(), boost::bind( &M4D::GUI::Viewer::RenderingExtension::render2DAlignedSlices, _1, aSliceIdx, aInterval, aPlane ) );
	}

	void
	preRender3D()
	{
		std::for_each( mExtensionList.begin(), mExtensionList.end(), boost::bind( &M4D::GUI::Viewer::RenderingExtension::preRender3D, _1 ) );
	}

	void
	postRender3D()
	{
		std::for_each( mExtensionList.begin(), mExtensionList.end(), boost::bind( &M4D::GUI::Viewer::RenderingExtension::postRender3D, _1 ) );
	}

	void
	addRenderingExtension( M4D::GUI::Viewer::RenderingExtension::Ptr aExtension )
	{
		mExtensionList.push_back( aExtension );
	}
protected:
	typedef std::list< M4D::GUI::Viewer::RenderingExtension::Ptr > ExtensionList;
	ExtensionList mExtensionList;
};



#endif /*PROXY_RENDERING_EXTENSION_H*/
