from typing import Callable
import torch

def test_positional_encoding(positional_encoding_fn: Callable[[torch.Tensor], torch.Tensor]):
    '''
    Test the PositionalEncoding class.
    '''
    test_pe_shape(positional_encoding_fn)
    test_pe_values(positional_encoding_fn)
    test_pe_forward(positional_encoding_fn)


def test_pe_shape(positional_encoding_fn: Callable[[torch.Tensor], torch.Tensor]):
    '''
    Test the shape of the PositionalEncoding.
    '''
    print("Testing Positional Encoding Shape ...")
    d_model = 16
    max_len = 50
    pe_layer     = positional_encoding_fn(d_model, max_len)
    input_tensor = torch.zeros((4, 10, d_model))
    output       = pe_layer(input_tensor)
    assert output.shape == input_tensor.shape, "Output shape does not match input shape"
    print("Test Passed: Positional Encoding Shape is Correct")


def test_pe_values(positional_encoding_fn: Callable[[torch.Tensor], torch.Tensor]):
    '''
    Test the values of the PositionalEncoding.
    '''
    print("Testing Positional Encoding Values ...")
    d_model = 4
    max_len = 10
    pe_layer     = positional_encoding_fn(d_model, max_len)
    expected_pe = torch.tensor(
       [[ 0.0000,  1.0000,  0.0000,  1.0000],
        [ 0.8415,  0.5403,  0.0100,  0.9999],
        [ 0.9093, -0.4161,  0.0200,  0.9998],
        [ 0.1411, -0.9900,  0.0300,  0.9996],
        [-0.7568, -0.6536,  0.0400,  0.9992],
        [-0.9589,  0.2837,  0.0500,  0.9988],
        [-0.2794,  0.9602,  0.0600,  0.9982],
        [ 0.6570,  0.7539,  0.0699,  0.9976],
        [ 0.9894, -0.1455,  0.0799,  0.9968],
        [ 0.4121, -0.9111,  0.0899,  0.9960]],
        dtype=pe_layer.pe.dtype
    )
    pe_buffer = pe_layer.pe.squeeze(0)[:max_len]
    assert torch.allclose(pe_buffer, expected_pe, rtol=1e-5, atol=1e-4), "Positional Encoding Values do not match expected values"
    print("Test Passed: Positional Encoding Values are Correct")


def test_pe_forward(positional_encoding_fn: Callable[[torch.Tensor], torch.Tensor]):
    '''
    Test the forward pass of the PositionalEncoding.
    '''
    print("Testing Positional Encoding Forward ...")
    d_model = 8
    max_len = 20
    pe_layer = positional_encoding_fn(d_model, max_len)
    input_tensor = torch.ones((2, 15, d_model))
    output = pe_layer(input_tensor)
    expected_output = input_tensor + pe_layer.pe[:, :input_tensor.size(1)]
    assert torch.allclose(output, expected_output, rtol=1e-5, atol=1e-5), "Positional Encoding Forward does not match expected values"
    print("Test Passed: Positional Encoding Forward is Correct")


def main():
    """
    Main function to run the positional encoding tests using the testing framework.
    """
    from hw4lib.model import PositionalEncoding
    from tests.testing_framework import TestingFramework

    framework = TestingFramework(
        test_categories={
            'PositionalEncoding': [
                {
                    'func': lambda: test_positional_encoding(PositionalEncoding),
                    'description': 'Test the positional encoding generation'
                }
            ]
        }
    )

    framework.run_tests()
    framework.summarize_results()

if __name__ == '__main__':
    main()