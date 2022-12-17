# To run unit tests: python -m unittest test

import unittest

from stats_tm.distribution import BinomialDist
from stats_tm.distribution import PoissonDist
from stats_tm.distribution import NormalDist
from stats_tm.distribution import StudentsTDist
from scipy import stats

class TestBinomialDist(unittest.TestCase):
    def setUp(self):
        self.binomial = BinomialDist(12, 0.7)

    def test_pmfcalculation(self): 
        myclass_result = round(self.binomial.calc_pmf(8, plot=False), 5)
        scipy_result = round(stats.binom.pmf(8, 12, 0.7), 5)
        self.assertAlmostEqual(myclass_result, scipy_result, None, 'incorrect probability mass function calculation for BinomialDist', 0.00001)

    def test_cumpcalculation(self): 
        myclass_result = round(self.binomial.calc_cum_p(10, plot=False), 5)
        scipy_result = round(stats.binom.cdf(10, 12, 0.7), 5)
        self.assertAlmostEqual(myclass_result, scipy_result, None, 'incorrect cumulative probability calculation for BinomialDist', 0.00001)

class TestBinomialDistSetted(unittest.TestCase):
    def setUp(self):
        self.binomial = BinomialDist(12, 0.7)
        self.binomial.n = 16
        self.binomial.p = 0.4

    def test_pmfcalculation(self):
        myclass_result = round(self.binomial.calc_pmf(8, plot=False), 5)
        scipy_result = round(stats.binom.pmf(8, 16, 0.4), 5)
        self.assertAlmostEqual(myclass_result, scipy_result, None, 'incorrect probability mass function calculation for setted BinomialDist', 0.00001)

    def test_cumpcalculation(self):
        myclass_result = round(self.binomial.calc_cum_p(10, plot=False), 5)
        scipy_result = round(stats.binom.cdf(10, 16, 0.4), 5)
        self.assertAlmostEqual(myclass_result, scipy_result, None, 'incorrect cumulative probability calculation for setted BinomialDist', 0.00001)
        
class TestPoissonDist(unittest.TestCase):
    def setUp(self):
        self.poisson = PoissonDist(5)

    def test_pmfcalculation(self):
        myclass_result = round(self.poisson.calc_pmf(8, plot=False), 5)
        scipy_result = round(stats.poisson.pmf(8, 5), 5)
        self.assertAlmostEqual(myclass_result, scipy_result, None, 'incorrect probability mass function calculation for PoissonDist', 0.00001)

    def test_cumpcalculation(self):
        myclass_result = round(self.poisson.calc_cum_p(3, plot=False), 5)
        scipy_result = round(stats.poisson.cdf(3, 5), 5)
        self.assertAlmostEqual(myclass_result, scipy_result, None, 'incorrect cumulative probability calculation for PoissonDist', 0.00001)
        
class TestPoissonDistSetted(unittest.TestCase):
    def setUp(self):
        self.poisson = PoissonDist(5)
        self.poisson.lambda_ = 9

    def test_pmfcalculation(self):
        myclass_result = round(self.poisson.calc_pmf(8, plot=False), 5)
        scipy_result = round(stats.poisson.pmf(8, 9), 5) 
        self.assertAlmostEqual(myclass_result, scipy_result, None, 'incorrect probability mass function calculation for setted PoissonDist', 0.00001)

    def test_cumpcalculation(self): 
        myclass_result = round(self.poisson.calc_cum_p(3, plot=False), 5)
        scipy_result = round(stats.poisson.cdf(3, 9), 5)
        self.assertAlmostEqual(myclass_result, scipy_result, None, 'incorrect cumulative probability calculation for setted PoissonDist', 0.00001)

class TestNormalDist(unittest.TestCase):
    def setUp(self):
        self.normal = NormalDist(10.3, 3.2)

    def test_pdfcalculation(self): 
        myclass_result = round(self.normal.calc_pdf(6.2, plot=False), 5)
        scipy_result = round(stats.norm.pdf(6.2, 10.3, 3.2), 5) 
        self.assertAlmostEqual(myclass_result, scipy_result, None, 'incorrect probability density function calculation for NormalDist', 0.00001)

    def test_cumpcalculation(self): 
        myclass_result = round(self.normal.calc_cum_p(7.7, plot=False), 5)
        scipy_result = round(stats.norm.cdf(7.7, 10.3, 3.2), 5) 
        self.assertAlmostEqual(myclass_result, scipy_result, None, 'incorrect cumulative probability calculation for NormalDist', 0.00001)
        
class TestNormalDistSetted(unittest.TestCase):
    def setUp(self):
        self.normal = NormalDist(10.3, 3.2)
        self.normal.mu = 4.5
        self.normal.sigma = 1.1

    def test_pdfcalculation(self): 
        myclass_result = round(self.normal.calc_pdf(6.2, plot=False), 5)
        scipy_result = round(stats.norm.pdf(6.2, 4.5, 1.1), 5) 
        self.assertAlmostEqual(myclass_result, scipy_result, None, 'incorrect probability density function calculation for setted NormalDist', 0.00001)

    def test_cumpcalculation(self): 
        myclass_result = round(self.normal.calc_cum_p(7.7, plot=False), 5)
        scipy_result = round(stats.norm.cdf(7.7, 4.5, 1.1), 5) 
        self.assertAlmostEqual(myclass_result, scipy_result, None, 'incorrect cumulative probability calculation for setted NormalDist', 0.00001)
        
class TestStudentTDist(unittest.TestCase):
    def setUp(self):
        self.studentT = StudentsTDist(2)

    def test_pdfcalculation(self): 
        myclass_result = round(self.studentT.calc_pdf(0.76, plot=False), 5)
        scipy_result = round(stats.t.pdf(0.76, 2), 5)
        self.assertAlmostEqual(myclass_result, scipy_result, None, 'incorrect probability density function calculation for StudentTDist', 0.00001)

    def test_cumpcalculation(self): 
        myclass_result = round(self.studentT.calc_cum_p(1.25, plot=False), 5)
        scipy_result = round(stats.t.cdf(1.25, 2), 5) 
        self.assertAlmostEqual(myclass_result, scipy_result, None, 'incorrect cumulative probability calculation for StudentTDist', 0.00001)
        
class TestStudentTDistSetted(unittest.TestCase):
    def setUp(self):
        self.studentT = StudentsTDist(2)
        self.studentT.v = 9

    def test_pdfcalculation(self): 
        myclass_result = round(self.studentT.calc_pdf(0.76, plot=False), 5)
        scipy_result = round(stats.t.pdf(0.76, 9), 5)
        self.assertAlmostEqual(myclass_result, scipy_result, None, 'incorrect probability density function calculation for setted StudentTDist', 0.00001)

    def test_cumpcalculation(self): 
        myclass_result = round(self.studentT.calc_cum_p(1.25, plot=False), 5)
        scipy_result = round(stats.t.cdf(1.25, 9), 5) 
        self.assertAlmostEqual(myclass_result, scipy_result, None, 'incorrect cumulative probability calculation for setted StudentTDist', 0.00001)
        
if __name__ == '__main__':
    unittest.main()