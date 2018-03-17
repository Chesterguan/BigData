import numpy as np

def sigmoid(x):
    """
    Compute the sigmoid function for the input here.
    """
    
    ### YOUR CODE HERE
    result = 1.0 / (1.0+np.exp(-x))
    ### END YOUR CODE
    
    return result

def sigmoid_grad(f):
    """
    Compute the gradient for the sigmoid function here. Note that
    for this implementation, the input f should be the sigmoid
    function value of your original input x. 
    """
    
    ### YOUR CODE HERE
    sg= f*(1-f)
    ### END YOUR CODE
    
    return sg

def test_sigmoid_basic():
    """
    Some simple tests to get you started. 
    Warning: these are not exhaustive.
    """
    print "Running basic tests..."
    x = np.array([[1, 2], [-1, -2]])
    f = sigmoid(x)
    g = sigmoid_grad(f)
    print f
    assert np.amax(f - np.array([[0.73105858, 0.88079708], 
        [0.26894142, 0.11920292]])) <= 1e-6
    print g
    assert np.amax(g - np.array([[0.19661193, 0.10499359],
        [0.19661193, 0.10499359]])) <= 1e-6
    print " these results verified!\n"

def test_sigmoid(): 
    """
    Use this space to test your sigmoid implementation by running:
        python q2_sigmoid.py 
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print "Running your tests..."
    ### YOUR CODE HERE
    x = np.array([[3, 4], [-3, -4]])
    f = sigmoid(x)
    g = sigmoid_grad(f)
    print f
    assert np.amax(f - np.array([[0.95257413, 0.98201379], 
        [0.04742587, 0.01798621]])) <= 1e-6
    print g
    assert np.amax(g - np.array([[0.04517666, 0.01766271],
        [0.04517666, 0.01766271]])) <= 1e-6
    g_ans=np.array([[0.04517666, 0.01766271],
        [0.04517666, 0.01766271]])
    assert np.allclose(g,g_ans,rtol=1e-06,atol=0) ## the result shoule be the same 
    print " personal test verified!\n"

    ### END YOUR CODE

if __name__ == "__main__":
    test_sigmoid_basic();
    test_sigmoid()
