from typing import Literal, Optional
import torch
    
# -----------------------------------------------------------------------------------------------------------------------------------
# The main tester: register this with the testing framework for dataset tests
def test_dataset_lm(dataset, method:Optional[Literal['__init__', '__get_item__', 'collate_fn']]=None):
    """
    Collection of test cases for the LM dataset.
    """ 
    # List of valid test methods    
    valid_args = ['__init__', '__get_item__', 'collate_fn']
    
    # If a specific method is requested
    if method:
        # Validate the method argument
        if method not in valid_args:
            raise ValueError(f"Invalid method: {method}. Valid methods are {valid_args}.")
        
        # Run the requested test
        if method == 'collate_fn':
            test_lm_data_collate_fn(dataset=dataset)  # Test batch collation
        elif method == '__get_item__':
            test_lm_data_getitem(dataset=dataset)     # Test item retrieval
        elif method == '__init__':
            test_lm_data_init(dataset=dataset)        # Test initialization
    
    # Otherwise run all tests       
    else:
        # Test dataset initialization and data loading
        test_lm_data_init(dataset=dataset)

        # Test single item retrieval functionality
        test_lm_data_getitem(dataset=dataset)
        
        # Test batch collation for training
        test_lm_data_collate_fn(dataset=dataset)


def test_lm_data_init(dataset):
    """
    Test case to validate the __init__ method of the LMDataset
    """
    print("Testing __init__ method ...")

    # Assert self.text_files is sorted
    assert dataset.text_files == sorted(dataset.text_files), "Text files are not sorted"
    print("Test Passed: Text files are sorted.")    

    # Assert alignment between transcripts
    assert len(dataset) == len(dataset.transcripts_shifted) == len(dataset.transcripts_golden), \
        "Shifted and golden transcripts are misaligned"
    print("Test Passed: Shifted and golden transcripts are aligned.")

    # Verify transcript decoding
    for shifted, golden in zip(dataset.transcripts_shifted, dataset.transcripts_golden):
        shifted_text = dataset.tokenizer.decode(shifted[1:])  # Exclude SOS token
        golden_text  = dataset.tokenizer.decode(golden[:-1])  # Exclude EOS token
        assert golden_text == shifted_text, \
            f"Decoded mismatch: {shifted_text} != {golden_text}"
    print("Test Passed: All transcripts are decoded correctly after removing SOS and EOS tokens.") 


def test_lm_data_getitem(dataset):
    """
    Test case to validate the __getitem__ method of the LMDataset
    """
    print("Testing __getitem__ method ...")

    for idx in range(min(len(dataset), 5)):  # Test the first 5 samples or fewer if dataset is small
        shifted_transcript, golden_transcript = dataset[idx]

        # Assert both are of type torch.LongTensor
        assert isinstance(shifted_transcript, torch.LongTensor), "Shifted transcript is not a torch.LongTensor"
        assert isinstance(golden_transcript, torch.LongTensor), "Golden transcript is not a torch.LongTensor"

        # Assert both have ndim 1
        assert shifted_transcript.ndim == 1, "Shifted transcript is not of shape (seq_len,)"
        assert golden_transcript.ndim == 1, "Golden transcript is not of shape (seq_len,)"
        
        # Validate transcript alignment
        assert shifted_transcript[0].item() == dataset.sos_token, f"Shifted transcript missing SOS token at index {idx}."
        assert golden_transcript[-1].item() == dataset.eos_token, f"Golden transcript missing EOS token at index {idx}."  

        shifted_decoded = dataset.tokenizer.decode(shifted_transcript[1:].tolist())  # Exclude SOS
        golden_decoded = dataset.tokenizer.decode(golden_transcript[:-1].tolist())  # Exclude EOS
        assert shifted_decoded == golden_decoded, \
            f"Decoded mismatch for sample {idx}: {shifted_decoded} != {golden_decoded}"
        
    print("Test Passed: SOS and EOS tokens are correctly placed for samples.")
    print("Test Passed: All transcripts are decoded correctly after removing SOS and EOS tokens.")


def test_lm_data_collate_fn(dataset):
    """
    Test case to validate the collate_fn method of the LMDataset
    """
    print("Testing collate_fn method ...")
    
    # Prepare a small batch from the dataset
    batch = [dataset[idx] for idx in range(min(len(dataset), 4))]  # Use up to 4 samples for testing
    batch_transcripts_pad, _, batch_lengths_transcripts = dataset.collate_fn(batch)

    # Validate transcript padding
    assert batch_transcripts_pad.ndim == 2, \
        f"Transcript batch has incorrect dimensions: Expected 2D tensor but got {batch_transcripts_pad.ndim}D tensor."
    print("Test Passed: Transcript batch has correct dimensions (2D tensor).")      
    
    # Validate length consistency
    assert len(batch_lengths_transcripts) == len(batch), \
        f"Transcript lengths mismatch: Expected {len(batch)}, but got {len(batch_lengths_transcripts)}."
    print("Test Passed: Transcript lengths are consistent with the batch size.")

    # Test for consistent padding
    max_length = torch.max(batch_lengths_transcripts)
    for transcript in batch_transcripts_pad:
        assert len(transcript) == max_length, "Inconsistent padding in batch."
    print("Test Passed: All sequences are padded to the same length.")

    # Test for correct padding values (assuming padding with zeros)
    padding_value = dataset.pad_token  # Replace with actual padding value if different
    for transcript, length in zip(batch_transcripts_pad, batch_lengths_transcripts):
        assert all(x == padding_value for x in transcript[length:]), \
            "Padding values are incorrect."
    print("Test Passed: Padding values are correct.")

    # Test for data type consistency
    assert isinstance(batch_transcripts_pad, torch.LongTensor), \
        "Batch transcripts are not of type torch.LongTensor."
    print("Test Passed: Batch transcripts are of correct data type.")


# -----------------------------------------------------------------------------------------------------------------------------------


def main():
    """
    Main function to run the dataset tests using the testing framework.
    """

    from hw4lib.data.lm_dataset import LMDataset
    from hw4lib.data.tokenizer import H4Tokenizer
    from tests.testing_framework import TestingFramework

    dummy_config = {
        'tokenization': {
            'token_type': "1k",       # [char, 1k, 5k, 10k]
            'token_map': {
                'char': './hw4lib/data/tokenizer_jsons/tokenizer_char.json',
                '1k'  : './hw4lib/data/tokenizer_jsons/tokenizer_1000.json',  
                '5k'  : './hw4lib/data/tokenizer_jsons/tokenizer_5000.json',
                '10k' : './hw4lib/data/tokenizer_jsons/tokenizer_10000.json'
            }
        },
        'data': {
            'root': "./hw4_data_subset/hw4p1_data",
            'train_partition': "train",
            'val_partition': "valid",
            'test_partition': "test",
            'subset': 1.0,
            'batch_size': 8,
            'NUM_WORKERS': 2
        }
    }
    
    tokenizer = H4Tokenizer(
        token_map  = dummy_config['tokenization']['token_map'], 
        token_type = dummy_config['tokenization']['token_type'], 
        validate   = False
    )

    train_dataset = LMDataset(
        partition = dummy_config['data']['train_partition'], 
        config    = dummy_config['data'],
        tokenizer = tokenizer
    )
    
    framework = TestingFramework(
        test_categories={
            'LMDataset': [
                {
                    'func': lambda : test_dataset_lm(train_dataset),
                    'description': 'Test the LMDataset class'
                }
            ]
        }
    )

    framework.run_tests()
    framework.summarize_results()

if __name__ == '__main__':
    main()
