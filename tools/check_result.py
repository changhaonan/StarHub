import os

if __name__ == "__main__":
    print("Checking result...")
    os.chdir("external/Easy3DViewer")
    os.system(f"python configure.py -d test_data -r http://localhost:8000")
    os.system("node app.js")