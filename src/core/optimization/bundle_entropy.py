import tensorflow as tf
import numpy as np
from . import projected_newton 




def BundleEntropyMethod(f, x, y, K):
    """Determine argmin_y f(x,y) over the n-dimensional unit-cube, for dim(y)=n.

    Necessary requirement: f is convex in the y argument. 
    Note: This is an approximate solution using subgradient descent in K iterations.
    We use a logarithmic boundary to enforce the solution within the unit cube. 

    Note: All calculations are done on float32 vectors

    Args:
        f: the objective function, arguments passed like f((x,y))
        x: Fixed x value of the model (Column Vector of shape (m, 1))
        y: Initial starting point (Column Vector of shape (n,1))
        K: Number of iterations        

    Returns:
        np.array of the same shape of y
    """
    # Make sure that only column vectors are passed for consistency
    assert x.shape[1] ==1 & y.shape[1] == 1, "You did not pass column vectors, the shape of x is {} and y is {}".format(x.shape, y.shape)

    # Initialize the lists
    G_l = []
    h_l = []

    # The iterations:
    for k in np.arange(K):
        print("This is iteration", k+1,"with current solution ", y)

        # Calculate the gradient w.r.t to y, ensuring that it is a tf.Variable
        if isinstance(y, np.ndarray):
            y_variable = tf.Variable(y, dtype="float32") # For tape.gradient to work, you cannot pass a tf.Tensor, it needs to be tf.Variable
            with tf.GradientTape() as tape:
                f_out = f((x,y_variable)) # Evaluate the gradient of f((x, . )) w.r.t. to the second argument at y 
            gradient = tape.gradient(f_out,y_variable) 
        else:
            with tf.GradientTape() as tape:
                f_out = f((x,y))    # Evaluate the gradient of f((x, . )) w.r.t. to the second argument at y 
            gradient = tape.gradient(f_out, y)




        #TODO: Delete the comment block
        # # Calculate the gradient of f w.r.t. to y
        # with tf.GradientTape() as tape:
        #     f_out = f((x,y))

        # # Evaluate the gradient of f((x, . )) w.r.t. to the second argument at y 
        # if isinstance(y, np.ndarray):
        #     y_variable = tf.Variable(y) # For tape.gradient to work, you cannot pass a tf.Tensor, it needs to be tf.Variable
        #     gradient = tape.gradient(f_out,y_variable)
        # else:
        #     gradient = tape.gradient(f_out, y)

        assert gradient.shape == y.shape,"Shape of gradient {} is not equal to the shape of ".format(gradient.shape) + " shape of input y {}".format(y.shape)

        # Safe the gradient in G and h
        # transpose the inputs
        gradient = tf.transpose(gradient)
        h_input = tf.transpose(f_out-tf.reduce_sum(tf.multiply(gradient, y))) # masking np.dot
        G_l.append(gradient[0]) # The 0 index is there to remove the outer bracket to reduce it from dim (1,n) -> (n,)
        h_l.append(h_input[0]) 

        # Note: At the moment we have list of row tensors, i.e. with shapes (n,) each

        # print("At iteration {} the h_l is {} and G_l ist {}". format(k+1, h_l, G_l))

        a_k = len(G_l)

        # Convert list to tf.Tensors, i.e. each element of the list becomes a row in a matrix 
        G = tf.stack(G_l)
        h = tf.stack(h_l)

        # print("Conversion yields G {} and h {}".format(G, h))

        # Solve for lambda using ProjNewtonLogistic
        # To send it to ProjeNewton Ligisitic we hace to convert the shape of h from (n,1) -> (n,)
        h = tf.transpose(h)[0]

        # print("Shape of h after conversion", h.shape)

        # To send it to ProjNewtonLogistic we have to convert the tensors to np.arrays
        G = G.numpy()
        h = h.numpy()

        # For the first iteration the value of lam is 1
        if a_k == 1:
            lam = np.array([[1.]])
        else:
            lam = projected_newton.ProjNewtonLogistic(G, h) # lam is a np.array of shape (n,)

        # now we need to convert lam to a column vector for the calculations below

        lam = lam.reshape((lam.shape[0],1))
        
        # Now solve for the new solution and substitute y
        # Cast it to tf.Variable to avoid type error

        # if k > 0:
        #     print("Solution of Proj Newton Logistic lam: ", lam, "with type", type(lam), "shape", lam.shape)


        y = tf.Variable(1/(1+np.exp(np.dot(np.transpose(G), lam))), dtype="float32")

        # Delete inactive constraints TODO: Why do we need to do this?
        G_l = [G_l[i] for i in np.arange(len(G_l)) if lam[i]>0]
        h_l = [h_l[i] for i in np.arange(len(h_l)) if lam[i]>0]
            
    return y

if __name__ == "__main__":
    import os
    print(os.getcwd())

    def f(arg):
        x,y = arg
        return 10.*(y[0]-0.5)**3 + 2.*((y[1]-0.5)*10.)**2 + x

    x = tf.Variable([[0.]], dtype="float32") #! This needs to be float 32 !
    y = tf.Variable([[0.7],[0.3]], dtype="float32")
    print("The input is x {} and y {} ".format(x, y))

    print("result", BundleEntropyMethod(f, x, y, 3))
