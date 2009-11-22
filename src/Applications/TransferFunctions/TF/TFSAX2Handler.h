#ifndef TF_SAXHANDLER
#define TF_SAXHANDLER

/*
 * http://xerces.apache.org/xerces-c/program-sax2-3.html
*/

#include <xercesc/sax2/DefaultHandler.hpp>

using namespace XERCES_CPP_NAMESPACE;

class MySAX2Handler : public DefaultHandler {
public:
	MySAX2Handler(){}
    void startElement(
        const   XMLCh* const    uri,
        const   XMLCh* const    localname,
        const   XMLCh* const    qname,
        const   Attributes&     attrs
    );
    void fatalError(const SAXParseException&);
};

#endif //TF_SAXHANDLER