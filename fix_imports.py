import glob
for file in glob.glob("tests/*.py"):
    if file == "tests/helpers.py": continue
    with open(file, "r") as f:
        content = f.read()
    if "from tests.helpers import get_test_config" not in content:
        content = "from tests.helpers import get_test_config\n" + content
    with open(file, "w") as f:
        f.write(content)
