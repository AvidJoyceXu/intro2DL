from typing import Literal, Optional
import torch

# -----------------------------------------------------------------------------------------------------------------------------------
# The main tester: register this with the testing framework for dataset tests
def test_dataset_asr(dataset, method:Optional[Literal['__init__', '__get_item__', 'collate_fn']]=None):
    """
    Collection of test cases for the ASR dataset.
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
            test_asr_data_collate_fn(dataset=dataset)  # Test batch collation
        elif method == '__get_item__':
            test_asr_data_getitem(dataset=dataset)     # Test item retrieval
        elif method == '__init__':
            test_asr_data_init(dataset=dataset)        # Test initialization
    
    # Otherwise run all tests       
    else:
        # Test dataset initialization and data loading
        test_asr_data_init(dataset=dataset)

        # Test single item retrieval functionality
        test_asr_data_getitem(dataset=dataset)
        
        # Test batch collation for training
        test_asr_data_collate_fn(dataset=dataset)


def test_asr_data_init(dataset):
    """
    Test case to validate the __init__ method of the ASRDataset
    """
    print("Testing __init__ method ...")

    # Assert dataset length matches FBANK files
    assert len(dataset) == len(dataset.fbank_files), "Dataset length mismatch with FBANK files."
    print("Test Passed: Dataset length matches FBANK files.")

    # Assert dataset length matches TRANSCRIPT files
    if dataset.partition != 'test-clean':
        assert len(dataset) == len(dataset.text_files), "Dataset length mismatch with TRANSCRIPT files."
        print("Test Passed: Dataset length matches TRANSCRIPT files.")

        # Assert order alignment between the FBANK files and TRANSCRIPT files
        for fbank_file, text_file in zip(dataset.fbank_files, dataset.text_files):
            assert fbank_file.split('.') == text_file.split('.'), \
                f"FBANK file {fbank_file} and TRANSCRIPT file {text_file} are misaligned."
        print("Test Passed: Order alignment between FBANK files and TRANSCRIPT files is correct.")
    
        # Assert alignment between features and transcripts
        assert len(dataset.feats) == len(dataset.transcripts_shifted) == len(dataset.transcripts_golden), \
            "Feature and transcript arrays are not aligned in length."
        print("Test Passed: Alignment between features and transcripts is correct.")
        
    # Validate num_feats
    for feat in dataset.feats:
        assert feat.shape[0] == dataset.config['num_feats'], \
            f"Feature mismatch: Expected {dataset.config['num_feats']} features, but got {feat.shape[0]}."
    print("Test Passed: All features have the correct number of dimensions (num_feats).")
        
    # Verify transcript decoding
    if dataset.partition != 'test-clean':
        for shifted, golden in zip(dataset.transcripts_shifted, dataset.transcripts_golden):
            shifted_text = dataset.tokenizer.decode(shifted[1:])  # Exclude SOS token
            golden_text = dataset.tokenizer.decode(golden[:-1])  # Exclude EOS token
            assert golden_text == shifted_text, \
                f"Decoded transcript mismatch: {shifted_text} != {golden_text}"
        print("Test Passed: All transcripts are decoded correctly after removing SOS and EOS tokens.")


def test_asr_data_getitem(dataset):
    """
    Test case to validate the __getitem__ method of the ASRDataset
    """
    print("Testing __getitem__ method ...")
    
    for idx in range(min(len(dataset), 5)):  # Test the first 5 samples or fewer if dataset is small
        feat, shifted_transcript, golden_transcript = dataset[idx]
        
        # Assert feature is a torch.FloatTensor
        assert isinstance(feat, torch.FloatTensor), "Feature is not a torch.FloatTensor"
        
        # Validate feature dimensions
        assert feat.shape[0] == dataset.config['num_feats'], \
            f"Feature mismatch: Expected {dataset.config['num_feats']} features, but got {feat.shape[0]}."
        
        # Check transcript behavior based on partition
        if dataset.partition == 'test-clean':
            assert shifted_transcript is None, f"Shifted transcript is not None for 'test-clean' at index {idx}."
            assert golden_transcript is None, f"Golden transcript is not None for 'test-clean' at index {idx}."
            print(f"Test Passed: Transcripts are None for 'test-clean' at index {idx}.")
        else:
            # Assert transcripts are torch.LongTensor
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
            
    print("Test Passed: All samples have correct feature dimensions and transcript alignment.")


def test_asr_data_collate_fn(dataset):
    """
    Test case to validate the collate_fn method of the ASRDataset
    """
    print("Testing collate_fn method ...")
    
    # Prepare a small batch from the dataset
    batch = [dataset[idx] for idx in range(min(len(dataset), 4))]  # Use up to 4 samples for testing
    batch_output = dataset.collate_fn(batch)
    
    # Extract batched outputs
    batch_feats_pad, batch_shifted_pad, batch_golden_pad, feat_lengths, transcript_lengths = batch_output
    
    # Test for data type consistency
    assert isinstance(batch_feats_pad, torch.FloatTensor), "Batch features are not of type torch.FloatTensor"
    
    # Validate feature padding
    assert batch_feats_pad.ndim == 3, \
        f"Feature batch has incorrect dimensions: Expected 3D tensor but got {batch_feats_pad.ndim}D tensor."
    print("Test Passed: Feature batch has correct dimensions (3D tensor).")
    
    # Test for consistent padding
    max_feat_length = torch.max(feat_lengths)
    assert batch_feats_pad.shape[1] == max_feat_length, "Inconsistent feature padding in batch."
    print("Test Passed: All sequences are padded to the same length.")

    if dataset.partition == 'test-clean':
        assert batch_shifted_pad is None, "Shifted transcripts batch is not None for 'test-clean'."
        assert batch_golden_pad is None, "Golden transcripts batch is not None for 'test-clean'."
        assert transcript_lengths is None, "Transcript lengths are not None for 'test-clean'."
        print("Test Passed: Transcripts and lengths are correctly set to None for 'test-clean'.")
    else:
        assert isinstance(batch_shifted_pad, torch.LongTensor), "Batch shifted transcripts are not of type torch.LongTensor"
        assert isinstance(batch_golden_pad, torch.LongTensor), "Batch golden transcripts are not of type torch.LongTensor"
        
        # Test for consistent padding in transcripts
        max_transcript_length = torch.max(transcript_lengths)
        assert batch_shifted_pad.shape[1] == max_transcript_length, "Inconsistent transcript padding in batch."
        print("Test Passed: All transcripts are padded to the same length.")

        # Test for correct padding values
        padding_value = dataset.pad_token
        for transcript, length in zip(batch_shifted_pad, transcript_lengths):
            assert all(x == padding_value for x in transcript[length:]), "Padding values are incorrect."
        print("Test Passed: Padding values are correct.")


def main():
    """
    Main function to run the dataset tests using the testing framework.
    """
    from hw4lib.data import ASRDataset, H4Tokenizer
    from tests.testing_framework import TestingFramework

    dummy_config = {
        'tokenization': {
            'token_type': "1k",
            'token_map': {
                'char': './hw4lib/data/tokenizer_jsons/tokenizer_char.json',
                '1k': './hw4lib/data/tokenizer_jsons/tokenizer_1000.json',
                '5k': './hw4lib/data/tokenizer_jsons/tokenizer_5000.json',
                '10k': './hw4lib/data/tokenizer_jsons/tokenizer_10000.json'
            }
        },
        'data': {
            'root': "./hw4_data_subset/hw4p2_data",
            'subset': 1.0,
            'batch_size': 8,
            'NUM_WORKERS': 2,
            'num_feats': 80,
            'norm': 'global_mvn',
            'specaug': True,
            'specaug_conf': {
                'apply_time_mask': True,
                'apply_freq_mask': True,
                'num_freq_mask': 2,
                'num_time_mask': 2,
                'freq_mask_width_range': 10,
                'time_mask_width_range': 10
            }
        }
    }

    tokenizer = H4Tokenizer(
        token_map=dummy_config['tokenization']['token_map'],
        token_type=dummy_config['tokenization']['token_type'],
        validate=False
    )

    train_dataset = ASRDataset(
        partition='train-clean-100',
        config=dummy_config['data'],
        tokenizer=tokenizer,
        isTrainPartition=True,
        global_stats=None
    )

    # Get the computed global stats from training set
    global_stats = None
    if dummy_config['data']['norm'] == 'global_mvn':
        global_stats = (train_dataset.global_mean, train_dataset.global_std)

    test_dataset = ASRDataset(
        partition='test-clean',
        config=dummy_config['data'],
        tokenizer=tokenizer,
        isTrainPartition=False,
        global_stats=global_stats
    )

    framework = TestingFramework(
        test_categories={
            'ASRDataset Train': [
                {
                    'func': lambda: test_dataset_asr(train_dataset),
                    'description': 'Test a Train instance of ASRDataset class'
                }
            ],
            'ASRDataset Test': [
                {
                    'func': lambda: test_dataset_asr(test_dataset),
                    'description': 'Test a Test instance of ASRDataset class'
                }
            ]
        }
    )

    framework.run_tests()
    framework.summarize_results()


if __name__ == "__main__":
    main()
