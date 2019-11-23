import numpy as np

# From: https://github.com/locuslab/icnn/blob/master/lib/bundle_entropy_dual.py
def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


def logexp1p(x):
    """ Numerically stable log(1+exp(x))"""
    y = np.zeros_like(x)
    I = x > 1
    y[I] = np.log1p(np.exp(-x[I])) + x[I]
    y[~I] = np.log1p(np.exp(x[~I]))
    return y


def ProjNewtonLogistic(A: "np.ndarray (k,n)", b: "np.ndarray (k,)", lam0=None):
    """ minimize_{lam>=0, sum(lam)=1} -(A*1 + b)^T*lam + sum(log(1+exp(A^T*lam)))
    This is the negative of the maximization problem
        Args:
            A: Corresponds to the G Matrix
            b: Corresponds to the h Vector, but is not a column vector, it is a row vector
            lam0: Is the vector along which we solve the problem
        Returns:
            np.array()
    """
    n = A.shape[0]
    c = np.sum(A, axis=1) + b  # Sum along row
    e = np.ones(n)

    eps = 1e-12
    ALPHA = 1e-5
    BETA = 0.5

    if lam0 is None:
        lam = np.ones(n) / n
    else:
        lam = lam0.copy()

    for i in range(100):
        # compute gradient and Hessian of objective
        ATlam = A.T.dot(lam)
        z = 1 / (
            1 + np.exp(-ATlam)
        )  # TODO: Problem: At values of 35 and above for ATlam entries, this vector is just np.ones(2,) -> H is then zero
        f = -c.dot(lam) + np.sum(logexp1p(ATlam))
        g = -c + A.dot(z)  # -> correct
        H = (A * (z * (1 - z))).dot(A.T)  # -> correct

        # change of variables
        i = np.argmax(lam)
        y = lam.copy()
        y[i] = 1  # y = lam with 1 at 1
        e[i] = 0  # e = np.ones with 0 at i => Will be used for sums ofer i != i^-

        H_22 = H[1, 1] - 2 * H[1, i] + H[i, i]

        g0 = g - e * g[i]
        H0 = H - np.outer(e, H[:, i]) - np.outer(H[:, i], e) + H[i, i] * np.outer(e, e)

        # compute bound set and Hessian of free set
        I = (y <= eps) & (g0 > 0)
        I[i] = True
        # if np.linalg.norm(g0[~I]) < 1e-10:  # TWEAKING THIS
        #     print("Projected: Norm too small at", np.linalg.norm(g0[~I]))
        # return lam
        d = np.zeros(n)  # Corresponds to -p
        H0_ = H0[~I, :][:, ~I]
        try:
            d[~I] = np.linalg.solve(H0_, -g0[~I])  # Return x s.t. H0_ * x = -g0[~I]
        except:
            print("\n== b", b)
            print("\n=== A\n\n", A)
            print("\n=== H\n\n", H)
            print("\n=== H0\n\n", H0)
            print("\n=== H0_\n\n", H0_)
            print("\n=== z\n\n", z)
            print("\n=== iter: {}\n\n".format(i))
            raise

        # line search
        t = 1.0
        for _ in range(50):
            # This is after eq (98)
            y_n = np.maximum(
                y + t * d, 0
            )  # Repeat this until we found a valid stepsize
            y_n[i] = 1
            # Retransformation: y -> x lam corresponds to x_k
            lam_n = y_n.copy()
            lam_n[i] = 1.0 - e.dot(y_n)
            if lam_n[i] >= 0:
                break
            if t < 1e-10:
                return lam_n
            t *= BETA

        e[i] = 1.0
        lam = lam_n.copy()
    return lam


if __name__ == "__main__":
    G = np.array([[1.0, 2.0], [5.0, 2.0]])
    h = np.array([3, 2])
    print(ProjNewtonLogistic(G, h))
