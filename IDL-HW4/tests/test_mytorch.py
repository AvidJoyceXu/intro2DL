from .testing_framework import TestingFramework
from .test_mytorch_linear import test_linear
from .test_mytorch_softmax import test_softmax
from .test_mytorch_scaled_dot_product_attention import test_scaled_dot_product_attention
from .test_mytorch_multi_head_attention import test_multi_head_attention
import json

if __name__ == "__main__":

    # Define the rubric for the tests
    rubric_dict = {
        "Linear": 5,
        "Softmax": 5,
        "ScaledDotProductAttention": 20,
        "MultiHeadAttention": 20,
    }

    testing_framework = TestingFramework(
        test_categories={k:[] for k in rubric_dict.keys()}
    )

    # Register Linear Tests
    testing_framework.register_test_case("Linear", test_linear, "Linear Tests")

    # Register Softmax Tests
    testing_framework.register_test_case("Softmax", test_softmax, "Softmax Tests")

    # Register ScaledDotProductAttention Tests
    testing_framework.register_test_case("ScaledDotProductAttention", test_scaled_dot_product_attention, "ScaledDotProductAttention Tests")

    # Register MultiHeadAttention Tests
    testing_framework.register_test_case("MultiHeadAttention", test_multi_head_attention, "MultiHeadAttention Tests")

    # Run all tests
    testing_framework.run_tests()

    # Summarize results
    testing_framework.summarize_results()