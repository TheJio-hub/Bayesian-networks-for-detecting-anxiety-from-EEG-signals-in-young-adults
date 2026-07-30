import ast
import os
import sys

root = os.getcwd()
py_files = []
for dirpath, dirnames, filenames in os.walk(root):
    # skip hidden directories
    if any(part.startswith('.') for part in dirpath.split(os.sep)):
        continue
    for f in filenames:
        if f.endswith('.py'):
            py_files.append(os.path.join(dirpath, f))

modules = set()
for p in py_files:
    try:
        with open(p, 'r', encoding='utf-8') as fh:
            src = fh.read()
        tree = ast.parse(src, filename=p)
    except Exception:
        continue
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                name = n.name.split('.')[0]
                modules.add(name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                name = node.module.split('.')[0]
                modules.add(name)

modules_sorted = sorted(modules)
print(f"Found {len(modules_sorted)} unique top-level modules/packages:\n")
for m in modules_sorted:
    print(m)

print('\nChecking importability using interpreter: ' + sys.executable + '\n')
results = {}
for m in modules_sorted:
    try:
        __import__(m)
        results[m] = ('OK', '')
    except Exception as e:
        results[m] = ('FAIL', str(e))

ok = [m for m, (s, _) in results.items() if s == 'OK']
fail = [m for m, (s, _) in results.items() if s == 'FAIL']
print(f"\nImport OK: {len(ok)}\n")
for m in ok:
    print(m)
print(f"\nImport FAIL: {len(fail)}\n")
for m in fail:
    print(m + ': ' + results[m][1])

# write report
with open('import_check_report.txt', 'w', encoding='utf-8') as out:
    out.write('Interpreter: ' + sys.executable + '\n')
    out.write('Found modules: ' + str(len(modules_sorted)) + '\n')
    out.write('\nOK:\n')
    for m in ok:
        out.write(m + '\n')
    out.write('\nFAIL:\n')
    for m in fail:
        out.write(m + ' : ' + results[m][1] + '\n')

print('\nReport written to import_check_report.txt')
