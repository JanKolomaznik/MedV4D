#ifndef TF_WINDOW
#define TF_WINDOW

#include <map>

#include <QtGui/QWidget>
#include <QtGui/QDockWidget>
#include <QtGui/QMainWindow>
#include <QtGui/QFileDialog>
#include <QtGui/QMessageBox>
#include <QtGui/QMenuBar>
#include <QtGui/QKeySequence>
#include <QtGui/QKeyEvent>
#include <QtGui/QVBoxLayout>

#include <TFCreator.h>
#include <TFPaletteButton.h>

#include <TFCommon.h>
#include <TFColor.h>
#include <TFHistogram.h>
#include <TFAbstractHolder.h>
#include <TFIndexer.h>

#include "ui_TFPalette.h"

namespace M4D {
namespace GUI {

class TFPalette : public QMainWindow{

    Q_OBJECT

public:

	typedef boost::shared_ptr<TFPalette> Ptr;

	TFPalette(QMainWindow* parent);
    ~TFPalette();

	void setupDefault();

	void setDomain(const TF::Size domain);
	TF::Size getDomain();
	
	M4D::Common::TimeStamp getTimeStamp();

	//TF::MultiDColor<dim>::Map::Ptr getColorMap();

	bool setHistogram(TF::Histogram::Ptr histogram, bool adjustDomain = true);

	template<typename BufferIterator>
	bool applyTransferFunction(BufferIterator begin, BufferIterator end){

		if(activeEditor_ < 0) on_addButton_clicked();
		if(activeEditor_ < 0) exit(0);

		return palette_.find(activeEditor_)->second.holder->applyTransferFunction<BufferIterator>(begin, end);
	}

private slots:

    void close_triggered(TF::Size index);

	void on_addButton_clicked();
	void on_removeButton_clicked();

	void change_activeHolder(TF::Size index);

protected:

	void resizeEvent(QResizeEvent*);
	void closeEvent(QCloseEvent *);

private:	

	struct Editor{
		TFAbstractHolder* holder;
		TFPaletteButton* button;

		Editor():
			holder(NULL),
			button(NULL){
		}

		Editor(TFAbstractHolder* holder, TFPaletteButton* button):
			holder(holder),
			button(button){
		}
	};

	typedef std::map<TF::Size, Editor> HolderMap;
	typedef HolderMap::iterator HolderMapIt;

    Ui::TFPalette* ui_;
	QMainWindow* mainWindow_;
	QVBoxLayout* layout_;

	TF::Histogram::Ptr histogram_;
	TF::Size domain_;

	M4D::Common::TimeStamp lastChange_;
	bool activeChanged_;

	TF::Indexer indexer_;
	int activeEditor_;
	HolderMap palette_;

	TFCreator creator_;

	void addToPalette_(TFAbstractHolder* holder);
	void removeFromPalette_(const TF::Size index);

	void activateNext_(HolderMapIt it);
};

} // namespace GUI
} // namespace M4D

#endif //TF_WINDOW