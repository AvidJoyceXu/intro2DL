import torch
import numpy as np
from torch.nn import Linear as PytorchLinear
from mytorch.nn.linear import Linear as MytorchLinear

def test_linear():
    """
    Test the Linear layer's forward and backward passes
    """
    print("Testing Linear Layer ...")
    test_linear_forward()
    test_linear_backward()


def test_linear_forward():
    """
    Test the Linear layer's forward pass
    """
    
    # Set the seed for reproducibility
    torch.manual_seed(11785)
    num_tests = 5

    for _ in range(num_tests):
        # Generate random dimensions
        batch_size = np.random.randint(1, 10)
        in_features = np.random.randint(1, 50)
        out_features = np.random.randint(1, 50)
        seq_len = np.random.randint(1, 10)

        # Initialize both implementations
        pytorch_linear = PytorchLinear(in_features, out_features)
        mytorch_linear = MytorchLinear(in_features, out_features)

        mytorch_linear.init_weights(pytorch_linear.weight.detach().numpy(), pytorch_linear.bias.detach().numpy())

        # Test Forward on random input
        X = torch.randn(batch_size, seq_len, in_features)
        pytorch_out = pytorch_linear(X)
        mytorch_out = mytorch_linear.forward(X.numpy())

        # Check if the output is close to the expected output
        assert np.allclose(pytorch_out.detach().numpy(), mytorch_out, rtol=1e-5, atol=1e-5), \
            f"Linear Forward Test Failed for shape: batch_size={batch_size}, seq_len={seq_len}, in_features={in_features}, out_features={out_features}"

    print("Test Passed: Linear Forward")


def test_linear_backward():
    """
    Test the Linear layer's backward pass
    """
    
    # Set the seed for reproducibility
    torch.manual_seed(11785)
    np.random.seed(11785)
    num_tests = 5

    for _ in range(num_tests):
        # Generate random dimensions
        batch_size = np.random.randint(1, 10)
        in_features = np.random.randint(1, 50)
        out_features = np.random.randint(1, 50)
        seq_len = np.random.randint(1, 10)

        # Initialize both implementations
        pytorch_linear = PytorchLinear(in_features, out_features)
        mytorch_linear = MytorchLinear(in_features, out_features)
        
        # Copy weights from PyTorch to MyTorch implementation
        mytorch_linear.init_weights(pytorch_linear.weight.detach().numpy(), pytorch_linear.bias.detach().numpy())

        # Forward pass with random input
        X = torch.randn(batch_size, seq_len, in_features, requires_grad=True)
        pytorch_out = pytorch_linear(X)
        mytorch_out = mytorch_linear.forward(X.detach().numpy())

        # Backward pass
        pytorch_out.sum().backward()
        pytorch_dX = X.grad.detach()
        pytorch_dW = pytorch_linear.weight.grad.detach()
        pytorch_db = pytorch_linear.bias.grad.detach()

        # MyTorch backward pass with ones since gradient of sum() is 1
        mytorch_dX = mytorch_linear.backward(np.ones_like(mytorch_out))

        # Check if gradients match
        assert np.allclose(pytorch_dX.numpy(), mytorch_dX, rtol=1e-5, atol=1e-5), \
            f"Input gradients don't match for shape: batch_size={batch_size}, seq_len={seq_len}, in_features={in_features}, out_features={out_features}"
        assert np.allclose(pytorch_dW.numpy(), mytorch_linear.dLdW, rtol=1e-5, atol=1e-5), \
            f"Weight gradients don't match for shape: batch_size={batch_size}, seq_len={seq_len}, in_features={in_features}, out_features={out_features}"
        assert np.allclose(pytorch_db.numpy(), mytorch_linear.dLdb, rtol=1e-5, atol=1e-5), \
            f"Bias gradients don't match for shape: batch_size={batch_size}, seq_len={seq_len}, in_features={in_features}, out_features={out_features}"

    print("Test Passed: Linear Backward")
