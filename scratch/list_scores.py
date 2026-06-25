from pgmpy.structure_score import BaseStructureScore
from skbase.lookup import all_objects

scores = all_objects(
    object_types=BaseStructureScore,
    package_name="pgmpy.structure_score",
    return_names=True,
)
for name, cls in scores:
    tags = cls._tags if hasattr(cls, '_tags') else 'No tags'
    print(f"Name: {name}, Class: {cls.__name__}, Tags: {tags}")
