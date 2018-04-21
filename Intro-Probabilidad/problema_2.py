import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

def plot_boundaries(X_train, X_test, y_train, y_test, score, probability_func, h = .02, ax = None):
    X = np.vstack((X_test, X_train))
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    if ax is None:
        ax = plt.subplot(1, 1, 1)
    
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    
    Z = probability_func(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    
    cf = ax.contourf(xx, yy, Z, 50, cmap=cm, alpha=.8)
    plt.colorbar(cf, ax=ax)
    #plt.colorbar(Z,ax=ax)

    # Plot also the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
               edgecolors='k', s=100)
    # and testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
               edgecolors='k', alpha=0.6, s=200)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
            size=40, horizontalalignment='right')

    plt.show()


def train_and_plot(X, y, h=1):
    # separo en dos sets, uno para entrenamiento y otro para prueba
	X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)

	gnb = GaussianNB()
	gnb.fit(X_train, y_train)

	score_train = gnb.score(X_train, y_train)

	plot_boundaries(X_train = X_train, X_test = X_test, y_train = y_train, y_test = y_test, score = score_train, probability_func = gnb.predict_proba)

	return


########
# MAIN #
########

# datos de entrada
problema1 = 'datasets\student_admission.txt'
problema2 = 'datasets\chip_tests.txt'

data = np.genfromtxt(fname=problema2, delimiter=",")

y = data[:, -1] # Última columna --> Label
x = data[:, :-1] # Primeras dos columnas --> Features

train_and_plot(x,y)
