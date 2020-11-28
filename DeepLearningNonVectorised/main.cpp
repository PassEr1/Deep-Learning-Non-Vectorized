#include "exceptions.hpp"
#include "data_loader.hpp"
#include <iostream>

int main()
{
    try
    {   /*
        In this example we are laoding up to 29 lines of data set consits 10=8 collumns each.
        Each training example consists 4x4 matrix's cells (raw by raw, 16 cells.) and 2 predications.
        The predictions is either the chance of the matrix to has a signed diagonal (with ones) or an ortogonal sign. 

         */
     
        PDataCollection data = DataLoader::read_data(L"..\\data_set__diagonal_or_ortogonal.csv", 29, 18);
        
    }
    catch (MyException & e)
    {
        std::wcout << L"An error accoured. The error code is:" << e.get_error_code() << std::endl;
    }
    catch (std::ios_base::failure & e)
    {
        std::wcout << L"An error accoured while attempted to read data set file. " << e.what() << std::endl;
    }
    catch (std::exception & e)
    {
        std::wcout << L"An unknown error accoured " << e.what() << std::endl;
    }
}
/*
TODO:
1. implement a cost-function checker.
2. wrap the DataCollection so nobody can assign or aoveride 2 PDataCollection from diffrent sizes

** solved:
1. check whather need to update weights in parallel --> sort of... ( after the the batch).
2. biases wont we a regular neuron, but an inner value within each neuron.

*/
