#pragma once
#include "data_loader.hpp"

class BatchLoader final
{
public:
	BatchLoader(const PDataCollection pdata_set, uint32_t chunk_size);

public:
	BatchLoader() = delete;
	BatchLoader(const BatchLoader&) = delete;
	BatchLoader(BatchLoader&&) = delete;
	BatchLoader& operator=(const BatchLoader&) = delete;
	BatchLoader& operator=(BatchLoader&&) = delete;

public:
	bool is_next() const;

public:
	DataCollection get_next();

private:
	const PDataCollection _pdata_set;
	uint32_t _offset;
	uint32_t _chunk_size;
};
