#pragma once
#include <cstdint>
#include <vector>
#include <string>
#include <memory>

using DataSample = std::vector<double>;
using DataCollection = std::vector<DataSample>;
using PDataCollection = std::shared_ptr<DataCollection>;

namespace DataLoader {
	std::vector<double> extract_from_single_line(std::wstring line, uint32_t width);
	PDataCollection read_data(const std::wstring& file_name, uint32_t max_rows, uint32_t colls);
}
