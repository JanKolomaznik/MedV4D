#include <TFHolderInterface.h>

namespace M4D{
namespace GUI{
	
void TFHolderInterface::save(){
	
	if(fileName_.isEmpty()) fileName_ = QDir::currentPath().append("/").append(windowTitle());

	fileName_ = QFileDialog::getSaveFileName(this,
		QObject::tr("Save Transfer Function"),
		fileName_,
		QObject::tr("TF Files (*.tf)"));

	if (fileName_.isEmpty()) return;

	QFile file(fileName_);
	if (!file.open(QFile::WriteOnly | QFile::Text))
	{
		QMessageBox::warning(this,
			QObject::tr("Transfer Functions"),
			QObject::tr("Cannot write file %1:\n%2.")
			.arg(fileName_)
			.arg(file.errorString()));
		return;
	}
	
	TFXmlWriter::Ptr writer(new TFXmlWriter(&file));

	writer->writeDTD("TransferFunctionsFile");

	writer->writeStartElement("Editor");

		writer->writeAttribute("Predefined",
			TF::convert<TF::Types::Predefined, std::string>(structure_.predefined));
		writer->writeAttribute("Holder",
			TF::convert<TF::Types::Holder, std::string>(structure_.holder));
		writer->writeAttribute("Function",
			TF::convert<TF::Types::Function, std::string>(structure_.function));
		writer->writeAttribute("Painter",
			TF::convert<TF::Types::Painter, std::string>(structure_.painter));
		writer->writeAttribute("Modifier",
			TF::convert<TF::Types::Modifier, std::string>(structure_.modifier));

		saveData_(writer);

	writer->finalizeDocument();

	file.close();

	saved_ = true;
}

bool TFHolderInterface::close(){

	if(!saved_)
	{
		QMessageBox msgBox;
		msgBox.setIcon(QMessageBox::Warning);
		msgBox.setText("Transfer Function has been modified.");
		msgBox.setInformativeText("Do you want to save your changes?");
		msgBox.setStandardButtons(QMessageBox::Save | QMessageBox::Discard | QMessageBox::Cancel);
		msgBox.setDefaultButton(QMessageBox::Save);
		int ret = msgBox.exec();

		if(ret == QMessageBox::Cancel) return false;
		if(ret == QMessageBox::Save) save();
	}
	return true;
}

} // namespace GUI
} // namespace M4D