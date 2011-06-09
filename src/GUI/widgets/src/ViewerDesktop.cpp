

void
ViewerDesktop::setLayoutOrganization( int cols, int rows )
{
	unsigned newSize = cols * rows;
	unsigned viewersSize = mViewers.size();
	int difference = newSize - viewersSize;

	if ( difference >= 0 )
	{
		for ( int i = 0; i < difference; i++ ) 
		{
			ViewerInfo info;
			info.viewer = createViewer();
			mViewers.push_back( info );
		}
	}
	else
	{ //TODO test selected viewer
		for ( unsigned i = newSize; i < viewersSize; i++ ) {
			delete mViewers[i];
		}
		mViewers.resize( newSize );
	}

	QLayout *mainLayout = new QVBoxLayout;

	QSplitter *mainSplitter = new QSplitter();
	mainSplitter->setOpaqueResize( true );
	mainSplitter->setOrientation( Qt::Vertical );

	for ( unsigned i = 0; i < rows; i++ )
	{
		QSplitter *splitter = new QSplitter();
		splitter->setOpaqueResize( true );
		for ( unsigned j = 0; j < cols; j++ )
		{   
			QWidget *widget = mViewers[i * cols + j].viewer;
			widget->resize( widget->sizeHint() );
			splitter->addWidget( widget );
		}
		mainSplitter->addWidget( splitter );
	}

	mainLayout->addWidget( mainSplitter );

	setLayout( mainLayout );

	//layoutRows = rows;
	//layoutColumns = columns;
}

AGLViewer *
ViewerDesktop::createViewer()
{
	//TODO
	return new GeneralViewer();
}

