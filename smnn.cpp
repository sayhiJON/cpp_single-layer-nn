#include <vector>
#include <time.h>

using namespace std;

static vector<float>    training_data    { 25733, 25971, 26458, 26430, 24874, 25413, 25538, 24527, 22878, 23996, 24579, 25426 },
                        test_data        { 25916, 25703, 25928, 26143, 26592, 25325, 24815, 26004, 26599, 27219, 27198, 26287 };

static float            normalization_factor = 10000.0f,
                        learning_rate        = 0.08f;
                
static int              trials = 120;

//* prototypes
vector<float>   normalize_data      (vector<float>, float);
vector<float>   get_initial_thetas  (int, float);
vector<float>   compute_derivative  (vector<float>, vector<float>);
float           compute_empirical_risk (vector<float>, vector<float>);
void            gradient_descent    (vector<float>, vector<float>, float, int);

int main () {
    srand (static_cast <unsigned> (time (0)));

    vector<float>   normalized_data         = normalize_data (training_data, normalization_factor),
                    normalized_test_data    = normalize_data (test_data, normalization_factor);

    gradient_descent (normalized_data, normalized_test_data, learning_rate, trials);

    return 0;
}

vector<float> normalize_data (vector<float> data, float factor) {
    vector<float> normalized;

    for (int index = 0; index < data.size (); index++)
        normalized.push_back (data[index] / factor);

    return normalized;
}

vector<float> get_initial_thetas (int count, float factor) {
    vector<float> theta;

    for (int index = 0; index < count; index++) {
        //* create a sign variable to make the range between -1 and 1
        int sign = rand () % 2;
        float random = static_cast <float> (rand ()) / static_cast <float> (RAND_MAX);
        
        //* you could ternary it (sign) ? ---- but that would cause the guaranteed check + a guaranteed multiplication operation
        if (sign)
            random *= -1;

        theta.push_back (random * factor);
    }

    return theta;
}

void gradient_descent (vector<float> training_data, vector<float> test_data, float learning_rate, int trials) {
    vector<float>   theta               = get_initial_thetas (3, 0.01f),
                    training_error,
                    test_error;

    for (int trial = 0; trial < trials; trial++) {
        //* get the derivatives with respect to the weights
        vector<float> risk_derivative = compute_derivative (training_data, theta);

        //* update our weights for this trial
        for (int t = 0; t < theta.size (); t++)
            theta[t] -= (learning_rate * risk_derivative[t]);

        //* get our error for training and testing
        training_error.push_back (compute_empirical_risk (training_data, theta));
        test_error.push_back (compute_empirical_risk (test_data, theta));
    }

    //* output our results
    printf ("Final training error: %6.6f - Final testing error: %6.6f", training_error.back (), test_error.back ());
}

float compute_empirical_risk (vector<float> data, vector<float> theta) {
    float total = 0.0f;

    //* little redundant and not generalized to size n
    for (int index = 0; index < data.size () - 2; index++) {
        float   predicted_dji   = (theta[0] * data[index + 1]) + (theta[1] * data[index]) + theta[2],
                observed_dji    = data[index + 2],
                error_signal    = observed_dji - predicted_dji;

        total += (error_signal * error_signal);
    }

    return total / (data.size () - 2);
}

vector<float> compute_derivative (vector<float> data, vector<float> theta) {
    //* should probably generalize this to a theta of size n instead of 3 but for now will work for this purpose
    vector<float> derivative_sums { 0.0f, 0.0f, 0.0f };

    for (int index = 0; index < data.size () - 2; index++) {
        float   predicted_dji = (theta[0] * data[index + 1]) + (theta[1] * data[index]) + theta[2],
                observed_dji = data[index + 2],
                error_signal = observed_dji - predicted_dji,

                //* now get derivatives

                //* we're working with means squared sum which is the summation from i = 0 to k of (y_i - y_p)^2
                //* and we want the derivative with respect to y_p

                //* y_i is our observed value

                //* y_p is a function of our activation of the output unit which is the summation of input * weight for a given output unit (linear activation)
                //* and we want the derivative of the activation with respect to the weight

                //* that gives us -2 * (y_i - y_p) * summation(inputs)
                //* for each input unit
                d_theta1 = -2 * error_signal * data[index + 1],
                d_theta2 = -2 * error_signal * data[index],
                d_theta3 = -2 * error_signal;

        derivative_sums[0] += d_theta1;
        derivative_sums[1] += d_theta2;
        derivative_sums[2] += d_theta3;
    }

    //* average it
    for (int index = 0; index < derivative_sums.size (); index++)
        derivative_sums[index] /= data.size ();

    return derivative_sums;
}