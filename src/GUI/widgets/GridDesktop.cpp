#include "MedV4D/GUI/widgets/GridDesktop.h"

namespace M4D
{
namespace GUI
{


GridDesktop::GridDesktop( QWidget *parent ): Desktop( parent )
{
	QVBoxLayout *layout = new QVBoxLayout;

	QSplitter *splitter = new QSplitter(parent);
	QListView *listview = new QListView;
	QTreeView *treeview = new QTreeView;
	QTextEdit *textedit = new QTextEdit;
	splitter->addWidget(listview);
	splitter->addWidget(treeview);
	splitter->addWidget(textedit);

	layout->addWidget( splitter );
	setLayout(layout);

	splitter->setOpaqueResize(false);
}

GridDesktop::~GridDesktop()
{

}

void 
GridDesktop::SetDesktopLayout ( unsigned rows, unsigned columns )
{

}

} /*namespace GUI*/
} /*namespace M4D*/


