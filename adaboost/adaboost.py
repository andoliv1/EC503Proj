import numpy as np
from sklearn.tree import DecisionTreeClassifier as DTC

class AdaBst:

    def __init__(self, X, Y, tmax):

        self.tmax = tmax
        self.X = X
        self.Y = Y
        self.n = X.shape[0]
        self.smp_w = np.zeros(shape=(tmax, self.n))
        self.st = np.zeros(shape=tmax, dtype=object)
        self.st_w = np.zeros(shape=tmax)
        self.err = np.zeros(shape=tmax)


    def adaclassifier(self):


        self.smp_w[0] = np.ones(shape=self.n) / self.n

        for t in range(self.tmax):

            #1) initialize distribution based on 1/n if it is first iteration
            #or on error weighted predictions of last iteration of not first
            curr_smp_w = self.smp_w[t]
            #2) find and train weak learner
            stump = DTC(max_depth=1, max_leaf_nodes=2)
            stump = stump.fit(self.X, self.Y, sample_weight=curr_smp_w)
            #3) find predictions and error
            Ypred = stump.predict(self.X)
            errcurr = curr_smp_w[(Ypred != self.Y)]
            errcurr = sum(errcurr)
            alphat = 0.5*np.log((1 - errcurr) / errcurr)
            #calculating and renormalizing weight distribution for current stump for later visualization
            new_smp_w = (
                    curr_smp_w * np.exp(-alphat * self.Y * Ypred)
            )
            new_smp_w /= new_smp_w.sum()
            if t + 1 < self.tmax:
                self.smp_w[t + 1] = new_smp_w

            #store parameters of weak learners to be used in the aggregate decision making
            self.st[t] = stump
            self.st_w[t] = alphat
            self.err[t] = errcurr

        return self

    def predict(self, X):
        wlpreds = np.array([stump.predict(X) for stump in self.st])
        pred = np.sign(np.dot(self.st_w, wlpreds))
        return pred