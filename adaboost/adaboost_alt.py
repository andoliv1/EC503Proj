import numpy as np
from WeightedTree import WeightedTree


class AdaBst_Alt:

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
            # print("This is self.X")
            # print(self.X)
            # print(t)
            # 1) initialize distribution based on 1/n if it is first iteration
            # or on error weighted predictions of last iteration of not first
            curr_smp_w = self.smp_w[t]
            # 2) find and train weak learner
            weights = np.zeros(2)
            counter = 0
            # print("This is the official array")
            # print(self.smp_w[t])
            for i in self.Y:
                # print(self.smp_w[t][counter])
                if (i == 1):
                    weights[0] += self.smp_w[t][counter]
                    # print(weights)
                else:
                    weights[1] += self.smp_w[t][counter]
                    # print(weights)
                counter += 1

            # print("This is weights")
            # print(weights)

            stump = WeightedTree(None, 1, None, None, 0)
            stump = WeightedTree.make_tree(stump, self.X, self.Y, 0, 0, curr_smp_w)

            # print("This is dec bound")
            # print(stump.boundary)

            # 3) find predictions and error
            Ypred = WeightedTree.evaluate_data(tree=stump, data=self.X)
            # print(Ypred)
            # print(self.Y)
            # print("current weight")
            # print(curr_smp_w)
            errcurr = curr_smp_w[(Ypred != self.Y)]
            errcurr = sum(errcurr)
            # print("This is the error")
            # print(errcurr)
            alphat = 0.5 * np.log((1 - errcurr) / errcurr)
            # calculating and renormalizing weight distribution for current stump for later visualization
            new_smp_w = (
                    curr_smp_w * np.exp(-alphat * self.Y * Ypred)
            )
            new_smp_w /= new_smp_w.sum()
            if t + 1 < self.tmax:
                self.smp_w[t + 1] = new_smp_w
            # print("weight after")
            # if t + 1 < self.tmax:
                # print(self.smp_w[t + 1])

            # store parameters of weak learners to be used in the aggregate decision making
            self.st[t] = stump
            self.st_w[t] = alphat
            self.err[t] = errcurr

        return self

    def predict(self, X):
        wlpreds = np.array([WeightedTree.evaluate_data(tree=stump, data=X) for stump in self.st])
        pred = np.sign(np.dot(self.st_w, wlpreds))
        # print(pred)
        # print(self.Y)
        return pred


