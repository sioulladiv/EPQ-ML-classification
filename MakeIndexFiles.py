import os
base_dir = "Garbage classification12"
ClassDirs = []

for root, dirs, files in os.walk(base_dir):
    ClassDirs.extend(dirs)
    break  

if not os.path.exists("zero-indexed-files2.txt"):
    with open("zero-indexed-files2.txt", "w") as f:
        for i, directory in enumerate(ClassDirs):
            full_path = os.path.join(base_dir, directory)
            try:
                for root, dirs, files in os.walk(full_path):
                    if files:
                        for file in files:
                            file_path = os.path.join(directory, file)
                            f.write(f"{file_path} {i}\n")            
            except Exception as e:
                print(f"Error accessing {directory}: {e}")