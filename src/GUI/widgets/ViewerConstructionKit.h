#ifndef _VIEWER_CONSTRUCTION_KIT_H
#define _VIEWER_CONSTRUCTION_KIT_H



template< typename SupportWidget, typename PortInterface, template<typename> DataProcessor >
class ViewerConstructionKit: public DataProcessor<SupportWidget>, public AGUIViewer
{
public:
	QWidget* 
	CastToQWidget()
		{ return static_cast< SupportWidget * >( this ); }

protected:

private:

};


#endif /*_VIEWER_CONSTRUCTION_KIT_H*/
