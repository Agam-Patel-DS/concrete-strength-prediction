import os

def create_project_structure(base_path='.'):
    """
    Creates a modular structure for the Concrete Compressive Strength Prediction project.
    """
    # Define folder structure
    folders = [
        os.path.join(base_path, 'data', 'raw'),
        os.path.join(base_path, 'data', 'processed'),
        os.path.join(base_path, 'notebooks'),
        os.path.join(base_path, 'scripts'),
        os.path.join(base_path, 'models'),
        os.path.join(base_path, 'logs'),
        os.path.join(base_path, 'configs'),
        os.path.join(base_path, 'tests'),
        os.path.join(base_path, 'docs')
    ]

    # Define empty files to be created
    files = [
        os.path.join(base_path, 'README.md'),
        os.path.join(base_path, 'setup.py'),
        os.path.join(base_path, 'requirements.txt'),
        os.path.join(base_path, 'configs', 'config.yaml'),
        os.path.join(base_path, 'scripts', '__init__.py'),
        os.path.join(base_path, 'models', '__init__.py'),
        os.path.join(base_path, 'tests', '__init__.py')
    ]

    # Create folders
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"Created folder: {folder}")

    # Create empty files
    for file in files:
        open(file, 'w').close()
        print(f"Created file: {file}")

    print(f"Project structure created successfully at {base_path}")

if __name__ == "__main__":
    create_project_structure()
