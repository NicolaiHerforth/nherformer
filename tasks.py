from invoke_tasks.all import get_tasks

ns = get_tasks()
ns.configure({"lint_directories": ["nherformer/prod", "tests"]})
