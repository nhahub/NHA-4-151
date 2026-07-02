import ast
import os

scripts = [
    'adaptive_followup.py', 'admin_dashboard.py', 'check_q.py',
    'confidence_evaluator.py', 'config.py', 'context_manager.py',
    'cv_parser.py', 'graph.py', 'interview_state.py', 'jd_parser.py',
    'llm_config.py', 'nodes.py', 'phase1_data_layer.py', 'phase2_orchestration.py',
    'router.py', 'server.py', 'session_manager.py', 'smart_question_gen.py',
    'soft_skills_evaluator.py', 'stt_service.py', 'test_api.py',
    'tts_service.py', 'vad_service.py'
]

output = []

class DeepAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.calls = []
        self.returns = []
        self.assignments = []

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            self.calls.append(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            self.calls.append(node.func.attr)
        self.generic_visit(node)

    def visit_Return(self, node):
        if node.value:
            if isinstance(node.value, ast.Name):
                self.returns.append(node.value.id)
            elif isinstance(node.value, ast.Dict):
                keys = [k.value for k in node.value.keys if isinstance(k, ast.Constant)]
                self.returns.append(f"Dict({keys})")
            elif isinstance(node.value, ast.Tuple):
                self.returns.append("Tuple")
            else:
                self.returns.append(type(node.value).__name__)
        self.generic_visit(node)

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.assignments.append(target.id)
        self.generic_visit(node)

def analyze_function(node):
    analyzer = DeepAnalyzer()
    analyzer.visit(node)
    
    args = [a.arg for a in node.args.args]
    docstring = ast.get_docstring(node) or 'No docstring.'
    docstring = docstring.strip().replace('\n', ' ')
    
    info = []
    args_str = ", ".join(args)
    info.append(f"- **Function**: `{node.name}({args_str})`")
    info.append(f"  - *Docstring*: {docstring}")
    
    if analyzer.assignments:
        unique_assigns = list(set(analyzer.assignments))
        assigns_str = ", ".join(unique_assigns[:10])
        if len(unique_assigns) > 10:
            assigns_str += "..."
        info.append(f"  - *Creates variables*: {assigns_str}")
        
    if analyzer.calls:
        unique_calls = list(set(analyzer.calls))
        calls_str = ", ".join(unique_calls[:10])
        if len(unique_calls) > 10:
            calls_str += "..."
        info.append(f"  - *Calls functions/methods*: {calls_str}")
        
    if analyzer.returns:
        unique_returns = list(set(analyzer.returns))
        info.append(f"  - *Returns*: {', '.join(unique_returns[:5])}")
        
    return '\n'.join(info)

for script in scripts:
    output.append(f"## File: {script}")
    if not os.path.exists(script):
        output.append("  File not found.\n")
        continue
    
    try:
        with open(script, 'r', encoding='utf-8') as f:
            content = f.read()
            tree = ast.parse(content)
    except Exception as e:
        output.append(f"  Error: {e}\n")
        continue

    has_content = False
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            has_content = True
            docstring = ast.get_docstring(node) or 'No docstring.'
            docstring = docstring.strip().replace('\n', ' ')
            output.append(f"### Class: `{node.name}`")
            output.append(f"- *Docstring*: {docstring}")
            
            methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
            if methods:
                for m in methods:
                    output.append(analyze_function(m))
            output.append("")
            
        elif isinstance(node, ast.FunctionDef):
            has_content = True
            output.append(analyze_function(node))
            output.append("")
            
    if not has_content:
        output.append("  No classes or functions.\n")

with open('deep_docs_output.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(output))
print('Done!')
