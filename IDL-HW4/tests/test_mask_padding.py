from typing import Callable, Optional
import torch

def test_mask_padding(mask_gen_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
    '''
    Test the padded mask generation function.
    Args:   
        mask_gen_fn (pad_mask_fn (Callable[[torch.Tensor, torch.Tensor], torch.Tensor])): The function to generate the padded mask.
    '''

    print("Testing Padding Mask ...") 

    # -------------------------------------------------    
    # Test Casse for input_lengths
    # -------------------------------------------------

    batch_sizes = [1, 2, 4]
    seq_lengths = [2, 4, 8]
    feat_lens   = [0, 5, 10]
    input_lengths = [
        torch.tensor([1]),
        torch.tensor([2,3]),
        torch.tensor([1,3,4,7])
    ]
    # pad_idx = [1, 2, 4]

    # Expected Masks    
    expected_mask_input_length1 = torch.tensor([
        [False,  True]
    ])
    expected_mask_input_length2 = torch.tensor([
        [False, False,  True,  True],
        [False, False, False,  True]
    ])
    expected_mask_input_length3 = torch.tensor([
        [False,  True,  True,  True,  True,  True,  True,  True],
        [False, False, False,  True,  True,  True,  True,  True],
        [False, False, False, False,  True,  True,  True,  True],
        [False, False, False, False, False, False, False,  True]
    ])


    #-----------------------------------------------
    # Run through Test Cases for input lengths        
    #-----------------------------------------------
    masks_input_length = [expected_mask_input_length1, expected_mask_input_length2, expected_mask_input_length3]

    for (batch_size, seq_length, feat_len, input_len, expected_mask) in zip(batch_sizes, seq_lengths, feat_lens, input_lengths, masks_input_length):
        if feat_len == 0:
            input_tensor = torch.randn(batch_size, seq_length)
        else:   
            input_tensor = torch.randn(batch_size, seq_length, feat_len)
        mask = mask_gen_fn(input_tensor, input_lengths = input_len)
        assert torch.equal(mask, expected_mask), f"Padding mask generation failed for batch size {batch_size}, sequence length {seq_length}, feature length {feat_len}, and input length {input_len}."
    

    print("Test Passed: Padding mask generation is correct.")

def main():
    """
    Main function to run the padding mask tests using the testing framework.
    """
    from hw4lib.model.masks import PadMask
    from tests.testing_framework import TestingFramework

    framework = TestingFramework(
        test_categories={
            'PaddingMask': [
                {
                    'func': lambda: test_mask_padding(PadMask),
                    'description': 'Test the padding mask generation'
                }
            ]
        }
    )

    framework.run_tests()
    framework.summarize_results()

if __name__ == '__main__':
    main()