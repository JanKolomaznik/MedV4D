#include<TFHolderFactory.h>


TFActions TFHolderFactory::createMenuTFActions(QMenu *menu){

	TFActions actions;

	//adding transferfunction types to menu
	actions.push_back(new TFAction(menu, TFTYPE_SIMPLE));	
	//actions.push_back(new TFAction(menu, TFTYPE_MYTYPE));

	return actions;
}

TFAbstractHolder* TFHolderFactory::createHolder(TFWindowI* window, TFType &holderType){

	switch(holderType)
	{
		case TFTYPE_SIMPLE:
		{
			return new TFSimpleHolder(window);
		}
		case TFTYPE_UNKNOWN:
		default:
		{
			assert("unknown holder");
			break;
		}
	}
	return NULL;
}

TFAbstractHolder* TFHolderFactory::loadHolder(TFWindowI* window){
	
	QString fileName = QFileDialog::getOpenFileName(
		window,
		QObject::tr("Open Transfer Function"),
		QDir::currentPath(),
		QObject::tr("TF Files (*.tf *.xml)"));

	if (fileName.isEmpty()) return NULL;

	QFile qFile(fileName);

	if (!qFile.open(QFile::ReadOnly | QFile::Text)) {
		QMessageBox::warning(
			window,
			QObject::tr("Transfer Functions"),
			QObject::tr("Cannot read file %1:\n%2.").arg(fileName).arg(qFile.errorString()));
		return NULL;
	}

	TFTypeSwitcher switcher;
	TFType tfType = switcher.read(&qFile);
	qFile.close();

	qFile.open(QFile::ReadOnly | QFile::Text);
	TFAbstractHolder* loaded = createHolder(window, tfType);

	if(loaded)
	{
		if(!loaded->load_(qFile))
		{ 
			QMessageBox::warning(
				window,
				QObject::tr("TFXmlSimpleReader"),
				QObject::tr("Parse error in file %1").arg(fileName));
		}
	}

	qFile.close();
	return loaded;
}

TFType TFHolderFactory::TFTypeSwitcher::read(QIODevice* device){

	setDevice(device);

	while(!atEnd())
	{
		readNext(); 

		if(isEndElement())
		{
			break;
		}

		if (isStartElement() && (name() == "TransferFunction"))
		{
			return convert<std::string, TFType>(attributes().value("type").toString().toStdString());
		}
	}
	return TFTYPE_UNKNOWN;
}