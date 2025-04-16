from typing import Callable, Dict, Any
import traceback

'''
Example usage:
if __name__ == "__main__":

    # Create an empty or pre-filled dict of test categories
    test_categories = {
        "data_processing": [],
        "model": [],
        "decoding": [],
        "integration": [],
        "training": [],
        "evaluation": [],
    }

    # Create instance of TestingFramework
    framework = TestingFramework(test_catetogies=test_categories)
    
    # Register tests
    framework.register_test_case(
        category    = "data_processing", 
        test_func   = some_dataset_test_fn, 
        description = "Test Dataset implementation"
    )
    
    # Run all registered tests
    framework.run_tests()

    # Summarize results
    framework.summarize_results()
    
'''

class TestingFramework:
    """
    A flexible testing framework that organizes and runs test cases by category.
    
    The framework supports:
    - Organizing tests into predefined categories (data_processing, model, etc.)
    - Running individual test cases or entire categories
    - Detailed error reporting and test summaries
    - Easy test registration with descriptions
    """
    
    def __init__(self, test_categories=None):
        """
        Initialize the TestingFramework with dictionaries to hold test cases
        for various components.
        
        Args:
            test_categories Optional(Dict): Dictionary mapping category names to lists of test cases.
                                  If empty, default categories will be used.
        """
        # Initialize default test categories if none provided
        if test_categories is None:
            self.test_categories = {
                "data_processing": [], # Tests for data loading and processing
                "model": [],           # Tests for model architecture and components
                "decoding": [],        # Tests for inference and decoding logic
                "integration": [],     # Tests for end-to-end functionality
                "training": [],        # Tests for training loops and optimization
                "evaluation": [],      # Tests for metrics and evaluation methods
            }
        else:
            self.test_categories = test_categories
        # Store test results for reporting
        self.results = {}

    def register_test_case(self, category: str, test_func: Callable, description: str = ""):
        """
        Add a test case to the specified category.
        
        Args:
            category (str): The category of the test (e.g., "data_processing", "model")
            test_func (Callable): The test function to run
            description (str): A brief description of the test case
            
        Raises:
            ValueError: If the specified category doesn't exist
        """
        if category not in self.test_categories:
            raise ValueError(f"Unknown category '{category}'. Available categories: {list(self.test_categories.keys())}")
        self.test_categories[category].append({"func": test_func, "description": description})

    def run_tests(self, category:str=None):
        """
        Execute all test cases and store results in the results dictionary.
        
        Args:
            category (str, optional): Specific category to test. If None, runs all categories.
            
        Raises:
            ValueError: If specified category doesn't exist
        """
        if category:
            if category not in self.test_categories:
                raise ValueError(f"Unknown category '{category}'. Available categories: {list(self.test_categories.keys())}")
            else:
                self.__run_tests_category(category=category)
        else:
            for category in self.test_categories.keys():
                self.__run_tests_category(category=category)
    
    def summarize_results(self):
        """
        Print a summary of all test results, organized by category.
        Shows total tests passed vs total tests run for each category.
        """
        print("\n\033[95m" + "="*80)
        print(f"{'Test Summary':^80}")
        print("="*80 + "\033[0m")
        
        for category in self.results.keys():
            if len(self.results[category]) > 0:
                self.__summarize_results_category(category=category)

    def get_autoresults(self, rubric_dict: Dict[str, float]):
        """
        Return a dictionary of test results for all categories.
        The rubric_dict is a dictionary of weights for each category.
        """
        assert rubric_dict.keys() == self.test_categories.keys(), "Rubric dictionary must have the same keys as test categories"
        auto_results = {}
        for category in rubric_dict.keys():
            auto_results[category] = rubric_dict[category] * (1 if all(item["status"] == "PASSED" for item in self.results[category]) else 0)
        return {'scores': auto_results}
    
    ## Private Methods -----------------------------------------------------------------------------------------

    def __run_tests_category(self, category:str):
        """
        Run all tests for a specific category and record their results.
        
        Args:
            category (str): The category of tests to run
            
        Prints progress and results for each test case.
        Records PASSED/FAILED/ERROR status and error messages if applicable.
        """
        tests = self.test_categories[category]
        self.results[category] = []
        
        print("\n\033[95m" + "="*80)
        print(f"Running tests for category: {category}")
        print("-"*80 + "\033[0m\n")
        
        for idx, test in enumerate(tests):
            test_num = f"[{idx+1:02d}/{len(tests):02d}]"
            try:
                print(f"\033[94m{test_num:<10} Running:  {test['description']}\033[0m")
                test["func"]()
                self.results[category].append({"status": "PASSED", "description": test["description"]})
                print(f"\033[92m{test_num:<10} PASSED:   {test['description']}\033[0m\n")
            except AssertionError as e:
                self.results[category].append({"status": "FAILED", "description": test["description"], "error": str(e)})
                print(f"\033[91m{test_num:<10} FAILED:   {test['description']}")
                print(f"{' '*10} Error:    {str(e)}\033[0m\n")
            except Exception as e:
                self.results[category].append({"status": "ERROR", "description": test["description"], "error": str(e)})
                print(f"\033[91m{test_num:<10} ERROR:    {test['description']}")
                print(f"{' '*10} Error:    {str(e)}\033[0m\n")
                print(f"{' '*10} Traceback: {traceback.format_exc()}\033[0m\n")

    def __summarize_results_category(self, category):
        """
        Print a summary of test results for a specific category.
        
        Args:
            category (str): The category to summarize
            
        Prints the ratio of passed tests to total tests for the category.
        """
        results = self.results[category]
        passed = sum(1 for r in results if r["status"] == "PASSED")
        total = len(results)
        
        print(f"\033[93m{'Category:':<12} {category:<30}")
        print(f"{'Results:':<12} {passed}/{total} tests passed ({passed/total*100:.1f}%)\033[0m")
