import re
import ast
from typing import List, Dict, Any
import multiprocessing


# This is a helper function that will run in the separate process
def _wrapper_execute(code: str, function_name: str, args: Dict[str, Any], queue: multiprocessing.Queue):
    """
    A wrapper to execute the user code and put the result in a queue.
    This function is intended to be the target of a multiprocessing.Process.
    """
    try:
        # Safe built-ins for common problems
        safe_builtins = {
            "len": len, "range": range, "set": set, "max": max, "min": min,
            "list": list, "dict": dict, "str": str, "int": int, "float": float,
            "bool": bool, "abs": abs, "sum": sum, "sorted": sorted,
            "enumerate": enumerate, "zip": zip, "all": all, "any": any
        }
        
        exec_globals = {"__builtins__": safe_builtins}
        exec_locals = {}
        
        # Execute the user's code to define the function
        exec(code, exec_globals, exec_locals)
        
        # Get the function and call it
        if function_name not in exec_locals:
            raise ValueError(f"Function '{function_name}' not found in code")
        
        user_function = exec_locals[function_name]
        result = user_function(**args)
        
        # Put the successful result in the queue
        queue.put({"result": result})
        
    except Exception as e:
        # Put any execution error in the queue
        queue.put({"error": str(e)})


class CodeExecutor:
    """Safely execute user code against test cases"""
    
    def __init__(self, timeout=5):
        self.timeout = timeout
    
    def run_test_cases(self, user_code: str, test_cases: List[Dict]) -> List[Dict]:
        """
        Execute user code against test cases
        """
        function_name = self._extract_function_name(user_code)
        if not function_name:
            return [{"error": "Could not extract function name from code"}]
        
        results = []
        
        for test_case in test_cases:
            try:
                args = self._parse_input_args(test_case["input"])
                
                # *** MODIFIED: Call the method with timeout logic ***
                actual_output = self._execute_with_timeout(user_code, function_name, args)
                
                expected_output = self._parse_expected(test_case["expected"])
                passed = self._compare_outputs(actual_output, expected_output)
                
                results.append({
                    "name": test_case["name"],
                    "input": test_case["input"],
                    "actual_output": actual_output,
                    "expected_output": expected_output,
                    "passed": passed,
                    "test_category": test_case.get("tests", "unknown")
                })
                
            # *** MODIFIED: Specifically catch TimeoutError ***
            except TimeoutError as e:
                results.append({
                    "name": test_case["name"],
                    "input": test_case["input"],
                    "actual_output": None,
                    "expected_output": self._parse_expected(test_case["expected"]),
                    "passed": False,
                    "error": str(e),
                    "test_category": test_case.get("tests", "unknown")
                })
            except Exception as e:
                results.append({
                    "name": test_case["name"],
                    "input": test_case["input"],
                    "actual_output": None,
                    "expected_output": self._parse_expected(test_case["expected"]),
                    "passed": False,
                    "error": str(e),
                    "test_category": test_case.get("tests", "unknown")
                })
        
        return results

    def _execute_with_timeout(self, code: str, function_name: str, args: Dict[str, Any]) -> Any:
        """
        Execute code in a separate process with a timeout.
        """
        # A queue to share data between processes
        queue = multiprocessing.Queue()
        
        # Create the process
        p = multiprocessing.Process(
            target=_wrapper_execute, 
            args=(code, function_name, args, queue)
        )
        
        p.start()
        
        # Wait for the process to finish or for the timeout
        p.join(self.timeout)
        
        # If the process is still running after the timeout, terminate it
        if p.is_alive():
            p.terminate()
            p.join()  # Clean up the terminated process
            raise TimeoutError(f"Execution timed out after {self.timeout} seconds")
        
        # If the process finished, get the result from the queue
        if not queue.empty():
            result_dict = queue.get()
            if "error" in result_dict:
                raise RuntimeError(f"Code execution failed: {result_dict['error']}")
            return result_dict.get("result")
        
        # If the process died without putting anything in the queue (e.g., segfault)
        raise RuntimeError("Code execution failed without a specific error.")

    
    # --- Helper methods remain the same ---

    def _extract_function_name(self, code: str) -> str:
        """Extract function name from code using AST"""
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    return node.name
        except:
            match = re.search(r'def\s+(\w+)\s*\(', code)
            if match:
                return match.group(1)
        return None
    
    def _parse_input_args(self, input_str: str) -> Dict[str, Any]:
        """Parse input string like 's="abc"' or 'nums=[1,2,3], target=5'"""
        try:
            local_vars = {}
            exec(input_str, {"__builtins__": {}}, local_vars)
            return local_vars
        except Exception as e:
            raise ValueError(f"Could not parse input: {input_str}. Error: {e}")
    
    def _parse_expected(self, expected_str: str) -> Any:
        """Parse expected output string to comparable value"""
        try:
            return eval(expected_str, {"__builtins__": {}})
        except:
            return expected_str

    def _compare_outputs(self, actual: Any, expected: Any) -> bool:
        """Compare actual vs expected output with type flexibility"""
        try:
            if actual is None and expected is None:
                return True
            if actual is None or expected is None:
                return False
            if type(actual) != type(expected):
                # Handle cases like list vs tuple for sorted results
                if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
                    return sorted(actual) == sorted(expected)
                return str(actual) == str(expected)
            return actual == expected
        except:
            return False