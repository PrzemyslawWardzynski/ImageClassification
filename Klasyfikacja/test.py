# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 2: k-NN i Naive Bayes
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# ----------------- TEN PLIK MA POZOSTAC NIEZMODYFIKOWANY ------------------
# --------------------------------------------------------------------------

import pickle
from unittest import TestCase, TestSuite, TextTestRunner, makeSuite

import numpy as np

from content import (classification_error, estimate_a_priori_nb, estimate_p_x_y_nb,
                     hamming_distance, model_selection_knn, model_selection_nb, p_y_x_knn,
                     p_y_x_nb, sort_train_labels_knn)

with open('test_data.pkl', mode='rb') as f:
    TEST_DATA = pickle.load(f)


class TestRunner(TextTestRunner):
    def __init__(self):
        super(TestRunner, self).__init__(verbosity=2)

    def run(self):
        suite = TestSuite()
        return super(TestRunner, self).run(suite)


class TestSuite(TestSuite):
    def __init__(self):
        super(TestSuite, self).__init__()
        self.addTest(makeSuite(TestHamming))
        self.addTest(makeSuite(TestSortTrainLabelsKNN))
        self.addTest(makeSuite(TestPYXKNN))
        self.addTest(makeSuite(TestClassificationError))
        self.addTest(makeSuite(TestModelSelectionKNN))
        self.addTest(makeSuite(TestEstimateAPrioriNB))
        self.addTest(makeSuite(TestEstimatePXYNB))
        self.addTest(makeSuite(TestPYXNB))
        self.addTest(makeSuite(TestModelSelectionNB))


class TestHamming(TestCase):
    def test_hamming_distance(self):
        data = TEST_DATA['hamming_distance']
        dist_expected = data['Dist']

        dist = hamming_distance(data['X'], data['X_train'])

        self.assertEqual(np.shape(dist), (40, 50))
        np.testing.assert_equal(dist, dist_expected)


class TestSortTrainLabelsKNN(TestCase):
    def test_sort_train_labels_knn(self):
        data = TEST_DATA['sort_train_labels_KNN']
        y_sorted_expected = data['y_sorted']

        y_sorted = sort_train_labels_knn(data['Dist'], data['y'])

        self.assertTrue(np.shape(y_sorted), (40, 50))
        np.testing.assert_equal(y_sorted, y_sorted_expected)


class TestPYXKNN(TestCase):
    def test_p_y_x_knn(self):
        data = TEST_DATA['p_y_x_KNN']
        p_y_x_expected = data['p_y_x']

        p_y_x = p_y_x_knn(data['y'], data['K'])

        self.assertEqual(np.shape(p_y_x), (40, 4))
        np.testing.assert_almost_equal(p_y_x, p_y_x_expected)


class TestClassificationError(TestCase):
    def test_classification_error(self):
        data = TEST_DATA['error_fun']
        error_val_expected = data['error_val']

        error_val = classification_error(data['p_y_x'], data['y_true'])

        self.assertEqual(np.size(error_val), 1)
        self.assertAlmostEqual(error_val, error_val_expected)


class TestModelSelectionKNN(TestCase):
    def test_model_selection_knn_best_error(self):
        data = TEST_DATA['model_selection_KNN']
        error_best_expected = data['error_best']

        best_error, _, _ = model_selection_knn(data['Xval'], data['Xtrain'], data['yval'],
                                               data['ytrain'], data['K_values'])

        self.assertEqual(np.size(best_error), 1)
        self.assertAlmostEqual(best_error, error_best_expected)

    def test_model_selection_knn_best_k(self):
        data = TEST_DATA['model_selection_KNN']
        best_k_expected = data['best_K']

        _, best_k, _ = model_selection_knn(data['Xval'], data['Xtrain'], data['yval'],
                                           data['ytrain'], data['K_values'])

        self.assertEqual(np.size(best_k), 1)
        self.assertEqual(best_k, best_k_expected)

    def test_model_selection_knn_errors(self):
        data = TEST_DATA['model_selection_KNN']
        errors_expected = data['errors']

        _, _, errors = model_selection_knn(data['Xval'], data['Xtrain'], data['yval'],
                                           data['ytrain'], data['K_values'])

        self.assertEqual(np.shape(errors), (5,))
        np.testing.assert_almost_equal(errors, errors_expected)


class TestEstimateAPrioriNB(TestCase):
    def test_estimate_a_priori_nb(self):
        data = TEST_DATA['estimate_a_priori_NB']
        p_y_expected = data['p_y']

        p_y = estimate_a_priori_nb(data['ytrain'])

        self.assertEqual(np.shape(p_y), (4,))
        np.testing.assert_almost_equal(p_y, p_y_expected)


class TestEstimatePXYNB(TestCase):
    def test_estimate_p_x_y_nb(self):
        data = TEST_DATA['estimate_p_x_y_NB']
        p_x_y_expected = data['p_x_y']

        p_x_y = estimate_p_x_y_nb(data['Xtrain'], data['ytrain'], data['a'], data['b'])

        self.assertEqual(np.shape(p_x_y), (4, 20))
        np.testing.assert_almost_equal(p_x_y, p_x_y_expected)


class TestPYXNB(TestCase):
    def test_p_y_x_nb(self):
        data = TEST_DATA['p_y_x_NB']
        p_y_x_expected = data['p_y_x']

        p_y_x = p_y_x_nb(data['p_y'], data['p_x_1_y'], data['X'])

        self.assertEqual(np.shape(p_y_x), (40, 4))
        np.testing.assert_almost_equal(p_y_x, p_y_x_expected)


class TestModelSelectionNB(TestCase):
    def test_model_selection_nb_best_error(self):
        data = TEST_DATA['model_selection_NB']
        error_best_expected = data['error_best']

        error_best, _, _, _ = model_selection_nb(data['Xtrain'], data['Xval'], data['ytrain'],
                                                 data['yval'], data['a_values'], data['b_values'])

        self.assertEqual(np.size(error_best), 1)
        self.assertAlmostEqual(error_best, error_best_expected)

    def test_model_selection_nb_best_a(self):
        data = TEST_DATA['model_selection_NB']
        best_a_expected = data['best_a']

        _, best_a, _, _ = model_selection_nb(data['Xtrain'], data['Xval'], data['ytrain'],
                                             data['yval'], data['a_values'], data['b_values'])
        self.assertEqual(np.size(best_a), 1)
        self.assertEqual(best_a, best_a_expected)

    def test_model_selection_nb_best_b(self):
        data = TEST_DATA['model_selection_NB']
        best_b_expected = data['best_b']

        _, _, best_b, _ = model_selection_nb(data['Xtrain'], data['Xval'], data['ytrain'],
                                             data['yval'], data['a_values'], data['b_values'])

        self.assertEqual(np.size(best_b), 1)
        self.assertEqual(best_b, best_b_expected)

    def test_model_selection_nb_errors(self):
        data = TEST_DATA['model_selection_NB']
        errors_expected = data['errors']

        _, _, _, errors = model_selection_nb(data['Xtrain'], data['Xval'], data['ytrain'],
                                             data['yval'], data['a_values'], data['b_values'])

        self.assertEqual(np.shape(errors), (3, 3))
        np.testing.assert_almost_equal(errors, errors_expected)
