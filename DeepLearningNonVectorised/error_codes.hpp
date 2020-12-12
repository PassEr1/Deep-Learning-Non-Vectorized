#pragma once

enum ErrorCode
{
	SUCESS = 0,
	DATA_SET_FILE_FORMAT_ERROR,
	NUMBER_OF_INPUTS_TO_HYPOTHESIS_IS_WRONG,
	WEIGHTS_FORM_NOT_MATH,
	NETWORK_ARCHITECTURE_ERROR,
	VECTOR_SUBSTRACTION_LENGTHS_ERROR,
	SET_DELTA_OF_LAYER_VECTOR_SIZE_ERROR,
	ZERO_DELTA_FOR_LAYER_INDEX_ERROR,
	COMPUTE_LAYER_PARTIAL_DERIVATIVES_INDEX_ERROR,
	VECTOR_ADD_LENGTHS_ERROR,
	LOGORITHMIC_COST_FUNCTION_ONE_SAMPLE_PARAMETERS_LENGTH_ERROR
};
