#include "exceptions.hpp"

MyException::MyException(ErrorCode err)
:_error_code(err)
{}

ErrorCode MyException::get_error_code()
{
	return _error_code;
}
