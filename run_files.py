import subprocess

files = ["scripts/0_preprocess.py",
         "scripts/1_clustering.py",
         "scripts/2_description.py"
]

for file in files:
    subprocess.run(["python", file], check=True)
