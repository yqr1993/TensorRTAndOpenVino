#include"qrcode.h"


ImageScanner scanner;

extern "C" void initZbarConfig()
{
	cout << "scanner starting..." << endl;
	scanner.set_config(ZBAR_NONE, ZBAR_CFG_ENABLE, 1);
	cout << "scanner ok!" << endl;
}


extern "C" Symbol_target detectQR(cv::Mat opencv_frame, int width, int height, string target)
{
    cv::Mat imageGray;
    cvtColor(opencv_frame, imageGray, CV_RGB2GRAY);
    uchar* raw = (uchar*)imageGray.data;

    Image imageZbar(width, height, "Y800", raw, width * height);
    scanner.scan(imageZbar);
    Image::SymbolIterator symbol = imageZbar.symbol_begin();

    Symbol_target symbol_data;
    symbol_data.data = "";

    for (;symbol != imageZbar.symbol_end(); ++symbol)
    {
        string data = symbol->get_data();
        if (data == target)
        {
            std::vector<cv::Point2f> corner;
            corner.push_back(cv::Point2f(symbol->get_location_x(0), symbol->get_location_y(0)));
            corner.push_back(cv::Point2f(symbol->get_location_x(3), symbol->get_location_y(3)));
            corner.push_back(cv::Point2f(symbol->get_location_x(2), symbol->get_location_y(2)));
            corner.push_back(cv::Point2f(symbol->get_location_x(1), symbol->get_location_y(1)));

            cv::Point center((symbol->get_location_x(0) + symbol->get_location_x(1) + symbol->get_location_x(2) + symbol->get_location_x(3)) / 4,
                (symbol->get_location_y(0) + symbol->get_location_y(1) + symbol->get_location_y(2) + symbol->get_location_y(3)) / 4
            );
               
            symbol_data.data = data;
            symbol_data.corner = corner;
            symbol_data.center = center;
        }
    }

    return symbol_data;
}
