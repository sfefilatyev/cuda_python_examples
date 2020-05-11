from monte_carlo_integrator import MonteCarloIntegrator

if __name__ == '__main__':

    integral_tests = [('y =log(x)*_P2(sin(x))', 11.733 , 18.472, 8.9999), ('y = _R( 1 + sinh(2*x)*_P2(log(x)) )', .9, 4, .584977), ('y = (cosh(x)*sin(x))/ sqrt( pow(x,3) + _P2(sin(x)))', 1.85, 4.81,  -3.34553) ]

    for f, lo, hi, expected in integral_tests:
        mci = MonteCarloIntegrator(math_function=f, precision='d', lo=lo, hi=hi)
        print('The Monte Carlo numerical integration of the function\n \t f: x -> %s \n \t from x = %s to x = %s is : %s ' % (f, lo, hi, mci.definite_integral()))
        print('where the expected value is : %s\n' % expected)
