# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 2: k-NN i Naive Bayes
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

from __future__ import division
from collections import Counter
import numpy as np


N_OF_CLASSES = 4
def hamming_distance(X, X_train):
    """
    :param X: zbior porownwanych obiektow N1xD
    :param X_train: zbior obiektow do ktorych porownujemy N2xD
    Funkcja wyznacza odleglosci Hamminga obiektow ze zbioru X od
    obiexktow X_train. ODleglosci obiektow z jednego i drugiego
    zbioru zwrocone zostana w postaci macierzy
    :return: macierz odleglosci pomiedzy obiektami z X i X_train N1xN2
    """
    denseX = X.toarray()
    denseX_train = X_train.toarray()
    tp_denseX_train = np.transpose(denseX_train)
    return denseX.astype(int) @ (~tp_denseX_train).astype(int) + \
           (~denseX).astype(int) @ tp_denseX_train.astype(int)


def sort_train_labels_knn(Dist, y):
    """
    Funkcja sortujaca etykiety klas danych treningowych y
    wzgledem prawdopodobienstw zawartych w macierzy Dist.
    Funkcja zwraca macierz o wymiarach N1xN2. W kazdym
    wierszu maja byc posortowane etykiety klas z y wzgledem
    wartosci podobienstw odpowiadajacego wiersza macierzy
    Dist
    :param Dist: macierz odleglosci pomiedzy obiektami z X
    i X_train N1xN2
    :param y: wektor etykiet o dlugosci N2
    :return: macierz etykiet klas posortowana wzgledem
    wartosci podobienstw odpowiadajacego wiersza macierzy
    Dist. Uzyc algorytmu mergesort.
    """
    dist_arg_sorted = np.argsort(Dist, kind="mergesort", axis=1)

    return y[dist_arg_sorted]

    pass

def p_y_x_knn(y, k):
    """
    Funkcja wyznacza rozklad prawdopodobienstwa p(y|x) dla
    kazdej z klas dla obiektow ze zbioru testowego wykorzystujac
    klasfikator KNN wyuczony na danych trenningowych
    :param y: macierz posortowanych etykiet dla danych treningowych N1xN2
    :param k: liczba najblizszuch sasiadow dla KNN
    :return: macierz prawdopodobienstw dla obiektow z X
    """
    nearest_neighbours = y[:, :k]
    p_y_x_t = np.array(
        [(nearest_neighbours == m+1).sum(axis=1) / k
         for m in range(0, N_OF_CLASSES)])
    return p_y_x_t.T

    pass

def classification_error(p_y_x, y_true):
    """
    Wyznacz blad klasyfikacji.
    :param p_y_x: macierz przewidywanych prawdopodobienstw
    :param y_true: zbior rzeczywistych etykiet klas 1xN.
    Kazdy wiersz macierzy reprezentuje rozklad p(y|x)
    :return: blad klasyfikacji
    """

    flipped_pyx = np.fliplr(p_y_x)
    class_max = N_OF_CLASSES - np.argmax(flipped_pyx,axis=1)
    return (class_max != y_true).mean()

    pass

def model_selection_knn(Xval, Xtrain, yval, ytrain, k_values):
    """
    :param Xval: zbior danych walidacyjnych N1xD
    :param Xtrain: zbior danych treningowych N2xD
    :param yval: etykiety klas dla danych walidacyjnych 1xN1
    :param ytrain: etykiety klas dla danych treningowych 1xN2
    :param k_values: wartosci parametru k, ktore maja zostac sprawdzone
    :return: funkcja wykonuje selekcje modelu knn i zwraca krotke (best_error,best_k,errors), gdzie best_error to najnizszy
    osiagniety blad, best_k - k dla ktorego blad byl najnizszy, errors - lista wartosci bledow dla kolejnych k z k_values
    """

    Dist = hamming_distance(Xval,Xtrain)
    sorted_labels = sort_train_labels_knn(Dist, ytrain)
    best_error = 1
    best_k = -1
    errors = []
    for k in k_values:
        p_y_x = p_y_x_knn(sorted_labels, k);
        k_error = classification_error(p_y_x, yval)
        errors.append(k_error)
        if k_error < best_error:
            best_error = k_error
            best_k = k

    return best_error,best_k,errors





    pass

def estimate_a_priori_nb(ytrain):
    """
    :param ytrain: etykiety dla dla danych treningowych 1xN
    :return: funkcja wyznacza rozklad a priori p(y) i zwraca p_y - wektor prawdopodobienstw a priori 1xM
    """
    N = ytrain.shape[0]
    _, count = np.unique(ytrain, return_counts=True)
    return count/N
    pass

def estimate_p_x_y_nb(Xtrain, ytrain, a, b):
    """
    :param Xtrain: dane treningowe NxD
    :param ytrain: etykiety klas dla danych treningowych 1xN
    :param a: parametr a rozkladu Beta
    :param b: parametr b rozkladu Beta
    :return: funkcja wyznacza rozklad prawdopodobienstwa p(x|y) zakladajac, ze x przyjmuje wartosci binarne i ze elementy
    x sa niezalezne od siebie. Funkcja zwraca macierz p_x_y o wymiarach MxD.
    """

    X_train = Xtrain.A
    N, D = X_train.shape
    result = np.empty((4, D))

    for k in range(4):
        k_indexes = (ytrain == k+1)
        word_count = np.sum(X_train[k_indexes, :], axis=0)
        result[k, :] = (word_count + a - 1) / (k_indexes.sum() + a + b - 2)

    return result

def p_y_x_nb(p_y, p_x_1_y, X):
    """
    :param p_y: wektor prawdopodobienstw a priori o wymiarach 1xM
    :param p_x_1_y: rozklad prawdopodobienstw p(x=1|y) - macierz MxD
    :param X: dane dla ktorych beda wyznaczone prawdopodobienstwa, macierz NxD
    :return: funkcja wyznacza rozklad prawdopodobienstwa p(y|x) dla kazdej z klas z wykorzystaniem klasyfikatora Naiwnego
    Bayesa. Funkcja zwraca macierz p_y_x o wymiarach NxM.
    """
    denseX = X.toarray()
    N, D = X.shape


    result = np.empty((N, p_y.shape[0]))
    for n in range(N):
        result[n, :] = np.prod(np.power(p_x_1_y, denseX[n, :]) * np.power(1. - p_x_1_y, 1. - denseX[n, :]), axis=1) * p_y
        result[n, :] /= np.sum(result[n, :])
    return result



pass

def model_selection_nb(Xtrain, Xval, ytrain, yval, a_values, b_values):
    """
    :param Xtrain: zbior danych treningowych N2xD
    :param Xval: zbior danych walidacyjnych N1xD
    :param ytrain: etykiety klas dla danych treningowych 1xN2
    :param yval: etykiety klas dla danych walidacyjnych 1xN1
    :param a_values: lista parametrow a do sprawdzenia
    :param b_values: lista parametrow b do sprawdzenia
    :return: funkcja wykonuje selekcje modelu Naive Bayes - wybiera najlepsze wartosci parametrow a i b. Funkcja zwraca
    krotke (error_best, best_a, best_b, errors) gdzie best_error to najnizszy
    osiagniety blad, best_a - a dla ktorego blad byl najnizszy, best_b - b dla ktorego blad byl najnizszy,
    errors - macierz wartosci bledow dla wszystkich par (a,b)
    """

    p_y = estimate_a_priori_nb(ytrain)
    best_a = -1
    best_b = -1
    error_best = 1
    i,j = 0,0
    errors = np.empty((len(a_values), len(b_values)))
    for a in a_values:

        j = 0
        for b in b_values:
            p_x_y = estimate_p_x_y_nb(Xtrain,ytrain,a,b)
            p_y_x = p_y_x_nb(p_y, p_x_y, Xval)
            error = classification_error(p_y_x, yval)
            errors[i,j] = error
            j+=1
            if error < error_best:
                error_best = error
                best_a = a
                best_b = b
        i+=1
    return error_best, best_a, best_b, errors



    pass
