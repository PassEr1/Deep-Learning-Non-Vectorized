#include "data_loader.hpp"
#include "exceptions.hpp"
#include <iostream>
#include <fstream>
#include <sstream>


std::vector<double> DataLoader::extract_from_single_line(std::wstring line, uint32_t width)
{
    std::wistringstream stream(line);
    std::wstring token;

    std::vector<double> extracted_line;
    static const wchar_t CSV_DELIM  = L',';
    while (std::getline(stream, token, CSV_DELIM))
    {
        extracted_line.push_back(std::stod(token));
    }

    if (extracted_line.size() != width)
    {
        throw MyException(ErrorCode::DATA_SET_FILE_FORMAT_ERROR);
    }

    return extracted_line;
}

PDataCollection DataLoader::read_data(const std::wstring& file_name, uint32_t max_rows, uint32_t colls)
{
    DataCollection data;
    std::wifstream data_set_file(file_name);

    std::wstring line;
    while (std::getline(data_set_file, line) && data.size() < max_rows)
    {
        std::vector<double> features = extract_from_single_line(line, colls);
        data.emplace_back(features);
    }

    return std::make_shared<DataCollection>(data);
}
