from pgmpy.causal_discovery import GES
import inspect

lines = inspect.getsourcelines(GES._fit)[0]
for i in range(40):
    print(lines[i], end="")
