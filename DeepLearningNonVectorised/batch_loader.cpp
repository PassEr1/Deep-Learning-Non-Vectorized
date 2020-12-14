#include "batch_loader.hpp"
#include "exceptions.hpp"
static const uint32_t INIT_VALUE = 0;

BatchLoader::BatchLoader(const PDataCollection pdata_set, uint32_t chunk_size)
:_pdata_set(pdata_set),
_offset(INIT_VALUE),
_chunk_size(chunk_size)
{
	if (chunk_size > pdata_set->size())
	{
		throw MyException(ErrorCode::BATCH_LOADER_CHUNK_SIZE_TOO_BIG);
	}
}

bool BatchLoader::is_next() const
{
	return _offset < _pdata_set->size();
}

DataCollection BatchLoader::get_next()
{
	DataCollection::const_iterator from = _pdata_set->begin() + _offset;
	DataCollection::const_iterator to;
	if ((_offset + _chunk_size) >= _pdata_set->size())
	{
		to = _pdata_set->end();
	}
	else 
	{
		to = from + _chunk_size;
	}

	DataCollection new_batch(from, to);
	_offset += _chunk_size;

	return std::move(new_batch);
}
