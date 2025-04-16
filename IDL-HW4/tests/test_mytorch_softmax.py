import torch
import numpy as np
from torch.nn import Softmax as PytorchSoftmax
from mytorch.nn.activation import Softmax as MytorchSoftmax

def test_softmax():
    """
    Test the Softmax activation function's forward and backward passes
    """

    print("Testing Softmax ...")
    test_softmax_forward()
    test_softmax_backward()


def test_softmax_forward():
    """
    Test the Softmax activation function's forward pass
    """
    
    # Set the seed for reproducibility
    torch.manual_seed(11785)
    num_tests = 5
    dims_to_test = [0, 1, 2, -1, -2]

    for _ in range(num_tests):

        # Select a random dimension to test
        dim = np.random.choice(dims_to_test, replace=False)
        pytorch_softmax = PytorchSoftmax(dim=dim)
        mytorch_softmax = MytorchSoftmax(dim=dim)

        # Generate random shape
        batch_size    = np.random.randint(1, 10)
        input_dim     = np.random.randint(1, 10)
        embedding_dim = np.random.randint(1, 10)
        shape = (batch_size, input_dim, embedding_dim)

        # Test Forward on random input
        Z = torch.randn(shape)
        pytorch_A = pytorch_softmax(Z)
        mytorch_A = mytorch_softmax.forward(Z.numpy())

        # Check if the output is close to the expected output
        assert np.allclose(pytorch_A.numpy(), mytorch_A, rtol=1e-5, atol=1e-5), "Softmax Forward Test Failed for shape: {} and dim: {}".format(shape, dim)

    print("Test Passed: Softmax Forward")



def test_softmax_backward():
    """
    Test the Softmax activation function's backward pass
    """

    # Set the seed for reproducibility
    torch.manual_seed(11785)
    np.random.seed(11785)
    num_tests = 5
    dims_to_test = [0, 1, 2, -1, -2]

    for _ in range(num_tests):

        # Select a random dimension to test
        dim = np.random.choice(dims_to_test, replace=False)
        pytorch_softmax = PytorchSoftmax(dim=dim)
        mytorch_softmax = MytorchSoftmax(dim=dim)
        
        # Generate random shape
        batch_size    = np.random.randint(1, 10)
        input_dim     = np.random.randint(1, 10)
        embedding_dim = np.random.randint(1, 10)
        shape = (batch_size, input_dim, embedding_dim)

        # Forward pass with random input
        Z = torch.randn(shape, requires_grad=True)
        pytorch_A = pytorch_softmax(Z)
        mytorch_A = mytorch_softmax.forward(Z.detach().numpy())

        # Backward pass with random input
        pytorch_A.sum().backward()
        pytorch_dZ = Z.grad.detach()

        # Gradient of .sum() wrt to input is 1, so we can use it to test the backward pass of mytorch_softmax
        mytorch_dZ = mytorch_softmax.backward(np.ones(shape))

        # Check if the output is close to the expected output   
        assert np.allclose(pytorch_dZ.numpy(), mytorch_dZ, rtol=1e-5, atol=1e-5), "Softmax Backward Test Failed for shape: {} and dim: {}".format(shape, dim)

    print("Test Passed: Softmax Backward")


