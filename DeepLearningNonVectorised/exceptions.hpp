#pragma once
#include "error_codes.hpp"

class MyException
{
public:
	MyException(ErrorCode err);

public:
	MyException()							   = delete;
	MyException& operator=(const MyException&) = delete;

public:
	ErrorCode get_error_code();

private:
	ErrorCode _error_code;
};
