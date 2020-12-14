#include "exceptions.hpp"
#include "data_loader.hpp"
#include "math_utils.hpp"
#include "trainner.hpp"
#include"batch_loader.hpp"
#include <time.h>
#include <iostream>

void print_some_test_sample(NetworkFullyConnected& network, PDataCollection data)
{

    std::wcout << "*** this is a test prediction: \n";
    auto prediction = network.predict(data->at(0));
    for (double input : data->at(0)) { std::wcout << input << " "; }
    std::wcout << std::endl;
    std::wcout << prediction[0] << L" " << prediction[1] << std::endl;

}

int main()
{
    try
    {   /*
        In this example we are laoding up to 29 lines of data set consits 10=8 collumns each.
        Each training example consists 4x4 matrix's cells (raw by raw, 16 cells.) and 2 predications.
        The predictions is either the chance of the matrix to has a signed diagonal (with ones) or an ortogonal sign. 

         */
        static const uint32_t DATA_SAMPLES_COUNT = 29;
        static const uint32_t INPUT_NEURONS_COUNT = 16;
        static const uint32_t OUTPUT_NEURONS_COUNT = 2;
        PDataCollection data = DataLoader::read_data(
            L"..\\data_set__diagonal_or_ortogonal.csv",
            DATA_SAMPLES_COUNT, 
            INPUT_NEURONS_COUNT 
            + OUTPUT_NEURONS_COUNT);

        static const uint32_t LAYER_1_NEURONS_COUNT = 16;
        static const uint32_t LAYER_2_NEURONS_COUNT = 8;
        std::vector<uint32_t> architecture = {
            INPUT_NEURONS_COUNT,
            LAYER_1_NEURONS_COUNT,
            LAYER_2_NEURONS_COUNT,
            OUTPUT_NEURONS_COUNT };
        NetworkFullyConnected network(architecture, MathFunctions::sigmoid);

        static const double LEARNING_RATE = 0.1;
        static const double STOP_TRAINING_THRESHOLD = 0.5;
        static const uint32_t BATCH_SIZE = 1;
        static const uint32_t EPOCHS = 100;

        SGD_Trainner::train(network,
            LEARNING_RATE,
            STOP_TRAINING_THRESHOLD,
            BATCH_SIZE,
            EPOCHS, data);

        print_some_test_sample(network, data);

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
2. wrap the DataCollection so nobody can assign or override 2 PDataCollection from diffrent sizes
4. try to move most of the api funtion for Neuron into BaseNeuron;

*/
