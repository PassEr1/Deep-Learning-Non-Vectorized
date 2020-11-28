#include <iostream>
#include <fstream>
#include <sstream>
#include "load_data.hpp"

std::vector<double> LoadData::extract_features(const std::wstring& line, wchar_t delim)
{
    std::wistringstream stream(line);
    std::wstring token;

    std::vector<double> features;

    while(std::getline(stream, token, delim))
    {
        features.push_back(std::stod(token));
    }

    return features;
}

DataSetPtr LoadData::load_data_set()
{
    std::vector<DataSample> data_set;
    static const std::wstring NAME_OF_DATA_FILE = L"../data_set.txt";
    std::wifstream data_set_file(NAME_OF_DATA_FILE);

    std::wstring line;
    while (std::getline(data_set_file, line))
    {
        static const wchar_t DELIMETER_IN_DB_FILE = L' ';
        std::vector<double> features = extract_features(line, DELIMETER_IN_DB_FILE);
        data_set.emplace_back(features);
    }

    return std::make_shared<DataSet>(data_set);
}
