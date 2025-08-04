from typing import List, Dict


class ExecutionAnalyzer:
    @staticmethod
    def analyze_results(execution_results: List[Dict]) -> Dict:
        """
        Convert raw execution results into coaching insights
        
        Input: Raw CodeExecutor results list of:
        {
            "name": test_case["name"],
            "input": test_case["input"],
            "actual_output": None,
            "expected_output": self._parse_expected(test_case["expected"]),
            "passed": False,
            "error": str(e),
            "test_category": test_case.get("tests", "unknown")
        }

        Output: Summary of executions
        {
            "total_tests": total,
            "passed": total - len(failed), 
            "failed": len(failed),
            "pass_rate": f"{((total-len(failed))/total*100):.0f}%" if total > 0 else "0%",
            "failed_tests": {
                "not_matching": where actual and expected not matching,
                "error_reason1": count,
                "error_reason2": count,
                ...
            },
        }

        """
        total_tests = len(execution_results)
        failed_tests = [res for res in execution_results if not res.get("passed", True)]
        pass_rate = (total_tests - len(failed_tests)) / total_tests * 100 if total_tests > 0 else 0
        error_reasons = {}
        for res in failed_tests:
            if res.get("error"):
                error_reasons[res["error"]] = error_reasons.get(res["error"], 0) + 1
            elif res.get("actual_output") != res.get("expected_output"):
                error_reasons["not_matching"] = error_reasons.get("not_matching", 0) + 1
        insights = {
            "total_tests": total_tests,
            "passed": total_tests - len(failed_tests),
            "failed": len(failed_tests),
            "pass_rate": f"{pass_rate:.0f}%",
            "failed_tests": error_reasons
        }
        return insights

        