import re
import math
import cmath
from decimal import Decimal, getcontext

class LatexFormulaEvaluator:
    def __init__(self, formula, variables):
        self.formula = formula
        self.variables = variables

    def _sanitize_latex(self, formula_str):
        """
        将 LaTeX 符号转换为 Python 表达式，但不替换变量。
        此方法处理所有已知的 LaTeX 符号，使其符合 Python 语法。
        """
        sanitized = formula_str
        
        # 移除 $$...$$ 和 \text{} 这种显示格式符号
        sanitized = sanitized.replace('$$', '')
        sanitized = re.sub(r'\\text{(.*?)}', r'\1', sanitized)

        # 替换数学操作符和函数
        sanitized = sanitized.replace('\\cdot', '*')
        sanitized = sanitized.replace('\\times', '*')
        
        # 处理 \frac{}{}
        sanitized = re.sub(r'\\frac{(.*?)}{(.*?)}', r'(\1) / (\2)', sanitized)
        
        # 处理 \max(...) 和 \min(...)
        sanitized = re.sub(r'\\max\((.*?)\)', r'max(\1)', sanitized)
        sanitized = re.sub(r'\\min\((.*?)\)', r'min(\1)', sanitized)
        
        # 处理指数 e^x 和对数 log(x)
        sanitized = sanitized.replace('e^', 'math.exp')
        sanitized = sanitized.replace('log', 'math.log')
        
        # 修正变量名中的下划线和方括号，使其符合Python语法
        sanitized = sanitized.replace('E[R_p]', 'E_R_p')
        sanitized = sanitized.replace('E[R_m]', 'E_R_m')
        sanitized = sanitized.replace('Z_\\alpha', 'Z_alpha')
        sanitized = sanitized.replace('\\sigma_p', 'sigma_p')
        sanitized = sanitized.replace('\\beta_i', 'beta_i')
        
        
        sanitized = re.sub(r'([a-zA-Z0-9_]+)\s*\(', lambda m: m.group(1) + ' * (' if m.group(1) not in ['max', 'min', 'math.exp', 'math.log'] else m.group(0), sanitized)

        # 再次处理变量名之间的隐含乘法
        # sanitized = re.sub(r'([a-zA-Z0-9_]+)\s*([a-zA-Z0-9_]+)', r'\1 * \2', sanitized)
        
        # sanitized = re.sub(r'([a-zA-Z0-9_])\s*\(', r'\1 * (', sanitized)
        return sanitized

    def _replace_variables(self, sanitized_formula_str):
        """
        用提供的数值替换表达式中的变量名。
        """
        final_expression = sanitized_formula_str
        
        processed_variables = {}
        for var_name, value in self.variables.items():
            processed_name = var_name.replace('\\', '').replace('[', '_').replace(']', '').replace('^', '_')
            processed_variables[processed_name] = str(value)

        for var_name, value in processed_variables.items():
            final_expression = re.sub(r'\b' + re.escape(var_name) + r'\b', value, final_expression)
        
        return final_expression

    def evaluate(self):
        """
        主评估方法，串联所有步骤。
        """
        try:
            sanitized_formula = self._sanitize_latex(self.formula)

            if '=' in sanitized_formula:
                sanitized_formula = sanitized_formula.split('=', 1)[1].strip()

            final_expression = self._replace_variables(sanitized_formula)

            print(f"final_expression:{final_expression}")

            safe_globals = {"__builtins__": None, "math": math, "cmath": cmath, "max": max, "min": min}
            
            # 使用 Decimal 类型进行精确计算
            result = eval(final_expression, safe_globals, {})

            # 结果四舍五入到四位小数
            return Decimal(result).quantize(Decimal('0.0001'))

            # return round(result, 4)

        except Exception as e:
            print(f"Error evaluating formula: {e} | Original: {self.formula} | Final Expression: {final_expression}")
            return None