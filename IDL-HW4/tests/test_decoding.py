import torch
from typing import List, Tuple, Type
import argparse

## Helper Classes ------------------------------------------------------------
class SearchTree:
    """Represents a deterministic tree for beam/greedy search testing"""
    def __init__(self, token_id: int, score: float = 0.0):
        '''
        Initialize a new SearchTree object
        Args:
            token_id: The token ID to represent this node
            score:    The score for this node
        '''
        self.token_id = token_id
        self.score    = score
        self.children = []  # List of SearchTree objects
    
    def add_path(self, tokens: List[int], score: float):
        """Add a path to the tree with corresponding scores"""
        current = self
        for token in tokens:
            # Check if child already exists
            child = next((c for c in current.children if c.token_id == token), None)
            if not child:
                child = SearchTree(token)
                child.score = score
                current.children.append(child)
            current = child

    def pretty_print(self, tokenizer=None, depth=0, score=None):
        """
        Print a human-readable representation of the tree
        Args:
            tokenizer: Optional tokenizer to decode token IDs into text
            depth:    Current depth in tree (for indentation) 
            score:    Score for reaching this node
        """
        indent    = "  " * depth
        token_str = str(self.token_id)
        if tokenizer:
            token_str = f"{self.token_id:<5} ({tokenizer.decode([self.token_id])})"
        
        score_str = f", score={score:>8.3f}" if score is not None else " " * 15
        print(f"{indent}{token_str}{score_str}")
        
        # Sort children by score for prettier printing
        sorted_children = sorted(self.children, key=lambda x: -x.score)  # Sort by descending score
        for child in sorted_children:
            child.pretty_print(tokenizer, depth + 1, child.score)


def make_tree(tokenizer, paths: List[Tuple[List[int], float]]):
    '''
    Create a tree from a list of paths
    Args:
        tokenizer: The tokenizer to use for decoding
        paths:    A list of paths to add to the tree
    Returns:
        The root of the tree
    '''
    root = SearchTree(tokenizer.sos_id)
    for tokens, score in paths:
        root.add_path(tokens, score)
    return root 

class DeterministicScoreFn:
    """Score function using an explicit tree for beam/greedy search testing"""
    def __init__(self, trees: List[SearchTree], tokenizer):
        self.trees      = trees
        self.tokenizer  = tokenizer
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the scores for each batch item
        Args:
            x: The input tensor: (batch_size, seq_len)
        Returns:
            The scores tensor: (batch_size, seq_len, vocab_size)
        """
        # Initialize the scores tensor for each batch item  
        batch_size = x.size(0)
        seq_len    = x.size(1)
        scores     = torch.full((batch_size, seq_len, self.tokenizer.vocab_size), 1.0)
        
        # For each item in batch
        for batch_idx in range(batch_size):
            scores[batch_idx, -1, :] = self.update_scores(self.trees[batch_idx], x[batch_idx])
        
        return scores[:, -1, :].log_softmax(dim=-1)
    
    def update_scores_helper(self, root: SearchTree, x: torch.Tensor, scores: torch.Tensor):
        """
        Update the scores for the root token
        Args:
            root: The root of the tree
            x:    The input tensor: (seq_len)
            scores: The scores tensor: (vocab_size)
        """
        # If the root token id does not match the first token in x, return  
        if x.size(0) != 0 and root.token_id != x[0].item():
            return
        
        # If x is empty, update the scores for the root token   
        if x.size(0) == 0:
            scores[root.token_id] = root.score
            return
        
        # If the root has children, update the scores for each child
        if root.children:
            for child in root.children:
                self.update_scores_helper(child, x[1:], scores)
        else:
            scores[self.tokenizer.eos_id] = root.score
        return
    
    def update_scores(self, root: SearchTree, x: torch.Tensor):
        '''
        Update the scores for the root token
        Args:
            root: The root of the tree
            x:    The input tensor: (seq_len)
        Returns:
            The updated scores tensor: (vocab_size)
        '''
        # Initialize scores tensor with random values WITHIN the range of 1.0 to 100.0  
        scores = torch.ones((self.tokenizer.vocab_size), dtype=torch.float) 
        self.update_scores_helper(root, x, scores)
        return scores

## ----------------------------------------------------------------------------

def test_beam_search_single_batch(generator, tokenizer):
    """Test beam search single batch using an explicit tree structure"""
    print("Testing Single Batch Beam Search ...")

    # Define paths with explicit scores
    paths = [
        [
            (tokenizer.encode("HELLO WORLD[EOS]"),  1000),
            (tokenizer.encode("YELLOW WORLD[EOS]"),  999),
            (tokenizer.encode("MELLOW WORLD[EOS]"),  998),
        ],
    ]
    
    max_len = max(len(tokens) for path in paths for tokens, score in path)
    beam_width = max(len(path) for path in paths)
    
    # Create trees and score function
    trees = [make_tree(tokenizer, path) for path in paths]
    score_fn = DeterministicScoreFn(trees, tokenizer)
    
    # Initialize generator
    generator = generator(
        score_fn=score_fn,
        tokenizer=tokenizer,
        max_length=max_len+10,
        device="cpu"
    )
    
    # Test beam search
    x = torch.full((1, 1), tokenizer.sos_id, dtype=torch.long)
    seqs, scores = generator.generate_beam(x, beam_width=beam_width)
    
    # Verify results
    expected_texts = ["HELLO WORLD", "YELLOW WORLD", "MELLOW WORLD"]
    for i, expected in enumerate(expected_texts):
        processed_seq = generator.post_process_sequence(seqs[0, i], tokenizer)
        generated = tokenizer.decode(processed_seq.tolist(), skip_special_tokens=True)
        print(f"Beam {i:<2} : generated: {generated:<12} | expected: {expected:<12}")
        assert generated == expected, f"Beam {i} generated '{generated}' but expected '{expected}'"


def test_beam_search_multi_batch(generator, tokenizer):
    """Test beam search using multiple explicit tree structures"""
    print("Testing Multi Batch Beam Search ...")
    
    # Reference batch_paths from original test
    batch_paths = [
        # Batch 0
        [
            (tokenizer.encode("HELLO WORLD[EOS]"),  1000),
            (tokenizer.encode("YELLOW WORLD[EOS]"), 999),
            (tokenizer.encode("MELLOW WORLD[EOS]"), 998)
        ],
        # Batch 1
        [
            (tokenizer.encode("GOOD BYE[EOS]"),  1000),
            (tokenizer.encode("GREAT DAY[EOS]"),  999),
            (tokenizer.encode("GUD NIGHT[EOS]"),  998)
        ],
    ]

    max_len = max(len(tokens) for batch in batch_paths for tokens, score in batch)
    beam_width = max(len(path) for path in batch_paths)
    
    # Create trees for each batch
    trees = [make_tree(tokenizer, path) for path in batch_paths]
    score_fn = DeterministicScoreFn(trees, tokenizer)
    
    generator = generator(
        score_fn=score_fn,
        tokenizer=tokenizer,
        max_length=max_len+10,
        device="cpu"
    )
    
    # Test beam search with these trees
    x = torch.full((2, 1), tokenizer.sos_id, dtype=torch.long)
    seqs, scores = generator.generate_beam(x, beam_width=beam_width)
    
    # Expected results for each batch
    expected_batch_texts = [
        ["HELLO WORLD", "YELLOW WORLD", "MELLOW WORLD"],
        ["GOOD BYE", "GREAT DAY", "GUD NIGHT"],
    ]
    
    # Verify results for each batch
    for batch_idx, expected_texts in enumerate(expected_batch_texts):
        processed_seqs = generator.post_process_sequence(seqs[batch_idx], tokenizer)  # Process all beams at once
        for beam_idx, (processed_seq, expected) in enumerate(zip(processed_seqs, expected_texts)):
            generated = tokenizer.decode(processed_seq.tolist(), skip_special_tokens=True)
            print(f"Batch {batch_idx:<2} : Beam {beam_idx:<2} : generated: {generated:<12} | expected: {expected:<12}")
            assert generated == expected, f"Batch {batch_idx}, Beam {beam_idx} generated '{generated}' but expected '{expected}'"


def test_greedy_search_single_batch(generator, tokenizer):
    """Test greedy search single batch"""
    print("Testing Single Batch Greedy Search ...")

    # Define single path with explicit score
    paths = [
        (tokenizer.encode("HELLO WORLD[EOS]"), 1000),
        (tokenizer.encode("YELLOW WORLD[EOS]"), 999),
        (tokenizer.encode("MELLOW WORLD[EOS]"), 998),
    ]

    # Create tree using existing SearchTree
    tree = make_tree(tokenizer, paths)
    score_fn = DeterministicScoreFn([tree], tokenizer)

    generator = generator(
        score_fn=score_fn,
        tokenizer=tokenizer,
        max_length=len(paths[0][0]),
        device="cpu"
    )

    # Test greedy search
    x = torch.full((1, 1), tokenizer.sos_id, dtype=torch.long)
    seq, _ = generator.generate_greedy(x)

    # Verify results
    expected_text = "HELLO WORLD"
    processed_seq = generator.post_process_sequence(seq[0], tokenizer)
    generated = tokenizer.decode(processed_seq.tolist(), skip_special_tokens=True)
    print(f"Generated: {generated:<12} | Expected: {expected_text:<12}")
    assert generated == expected_text, f"Generated '{generated}' but expected '{expected_text}'"


def test_greedy_search_multi_batch(generator, tokenizer):
    """Test greedy search with multiple batches"""
    print("Testing Multi Batch Greedy Search ...")

    # Define paths for each batch
    batch_paths = [
        [(tokenizer.encode("HELLO WORLD[EOS]"), 1000), (tokenizer.encode("YELLOW WORLD[EOS]"), 999), (tokenizer.encode("MELLOW WORLD[EOS]"), 998)],
        [(tokenizer.encode("GOOD BYE[EOS]"), 1000), (tokenizer.encode("GREAT DAY[EOS]"), 999), (tokenizer.encode("GUD NIGHT[EOS]"), 998)]
    ]

    # Create trees using existing SearchTree
    trees = [make_tree(tokenizer, path) for path in batch_paths]
    score_fn = DeterministicScoreFn(trees, tokenizer)
    
    generator = generator(
        score_fn=score_fn,
        tokenizer=tokenizer,
        max_length=max(len(path[0][0]) for path in batch_paths),
        device="cpu"
    )

    # Test greedy search
    x = torch.full((2, 1), tokenizer.sos_id, dtype=torch.long)
    seq, _ = generator.generate_greedy(x)

    # Verify results
    expected_texts = ["HELLO WORLD", "GOOD BYE"]
    for batch_idx, expected in enumerate(expected_texts):
        processed_seq = generator.post_process_sequence(seq[batch_idx], tokenizer)
        generated = tokenizer.decode(processed_seq.tolist(), skip_special_tokens=True)
        print(f"Batch {batch_idx:<2} : Generated: {generated:<12} | Expected: {expected:<12}")
        assert generated == expected, f"Batch {batch_idx} generated '{generated}' but expected '{expected}'"


def test_greedy_decoding(generator, tokenizer):
    """Run all greedy search tests"""
    # Greedy Search Tests 
    test_greedy_search_single_batch(generator, tokenizer)
    test_greedy_search_multi_batch(generator, tokenizer)

def test_beam_decoding(generator, tokenizer):
    """Run all beam search tests"""
    # Beam Search Tests 
    test_beam_search_single_batch(generator, tokenizer)
    test_beam_search_multi_batch(generator, tokenizer)

def test_decoding(generator, tokenizer):
    """Run all decoding tests"""
    print("Testing Decoding ...")
    
    test_greedy_search_single_batch(generator, tokenizer)
    test_greedy_search_multi_batch(generator, tokenizer)
    test_beam_search_single_batch(generator, tokenizer)
    test_beam_search_multi_batch(generator, tokenizer)


def main():
    """Main function to run the tests"""
    from hw4lib.decoding import SequenceGenerator
    from hw4lib.data import H4Tokenizer
    from tests.testing_framework import TestingFramework

    # Add argument parsing
    parser = argparse.ArgumentParser(description='Run decoding tests')
    parser.add_argument('--mode', choices=['all', 'greedy', 'beam'], default='all',
                      help='Specify which decoding tests to run (default: all)')
    args = parser.parse_args()

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
    }
    
    tokenizer = H4Tokenizer(
        token_map  = dummy_config['tokenization']['token_map'], 
        token_type = dummy_config['tokenization']['token_type'], 
        validate   = False
    )

    # Configure tests based on mode argument
    test_categories = {
        'Decoding': []
    }

    if args.mode in ['all', 'greedy']:
        test_categories['Decoding'].append({
            'func': lambda: test_greedy_decoding(SequenceGenerator, tokenizer),
            'description': 'Test greedy decoding'
        })
    
    if args.mode in ['all', 'beam']:
        test_categories['Decoding'].append({
            'func': lambda: test_beam_decoding(SequenceGenerator, tokenizer),
            'description': 'Test beam decoding'
        })

    framework = TestingFramework(test_categories=test_categories)
    framework.run_tests()
    framework.summarize_results()

if __name__ == '__main__':
    main()

