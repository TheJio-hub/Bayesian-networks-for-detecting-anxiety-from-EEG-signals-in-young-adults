from pgmpy.causal_discovery import GES
import inspect

# Imprimir clases base
print("Clases base de GES:")
print(GES.__mro__)

# Imprimir metodos heredados
ges = GES()
print("\nMetodos en ges:")
print([m for m in dir(ges) if not m.startswith('__')])
