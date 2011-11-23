#ifndef PYTHON_TERMINAL_H
#define PYTHON_TERMINAL_H

#ifdef USE_PYTHON



#include <boost/python.hpp>

#include "MedV4D/Common/PythonInterpreter.h"
#include "GUI/widgets/TerminalWidget.h"

namespace M4D
{
namespace GUI
{


class PythonTerminal: public TerminalWidget
{
	Q_OBJECT;
public:
	PythonTerminal( QWidget *aParent = NULL ): TerminalWidget( aParent )
	{
		mInterpreter.setStdPrinter( Printer::Ptr( new TerminalStdPrinter( this ) ) );
		mInterpreter.setErrPrinter( Printer::Ptr( new TerminalErrPrinter( this ) ) );
	}

protected:
	struct TerminalStdPrinter: public Printer
	{
		TerminalStdPrinter( PythonTerminal *aTerminal ): mTerminal( aTerminal ) 
		{
			ASSERT( aTerminal );
		}

		void
		operator()( const std::string &aText )
		{
			if( aText.empty() ) {
				return;
			}
			mTerminal->appendText( aText.data() );
		}

		PythonTerminal *mTerminal;
	};

	struct TerminalErrPrinter: public Printer
	{
		TerminalErrPrinter( PythonTerminal *aTerminal ): mTerminal( aTerminal ) 
		{
			ASSERT( aTerminal );
		}

		void
		operator()( const std::string &aText )
		{
			if( aText.empty() ) {
				return;
			}
			mTerminal->appendErrorText( aText.data() );
		}

		PythonTerminal *mTerminal;
	};

	void
	processInput( const QString &str ) 
	{
		mInterpreter.exec( str.toStdString() );
	}

	PythonInterpreter mInterpreter;	
};

}//GUI
}//M4D

#endif /*USE_PYTHON*/

#endif /*PYTHON_TERMINAL_H*/
