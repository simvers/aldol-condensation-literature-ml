
files = ["0_preprocess.py",
         "1_clustering.py",
         "2_description.py"
]

for file in files:
    exec(open(file).read())
