import subprocess
import os

root = "../../results/out"
folders = ['4', '8', '16', '32']
sizes = ['4', '16', '32', '64', '128', '256', '512', '1024', '2048', '4096', '8192']
types = ['dense']
output_root = "../../results/reconstruct"

def resurrect_file(in_name, out_name, size):
    if os.path.isfile(out_name):
        print(f"MATRIX FILE {out_name} ALREADY EXISTS ; SKIPPING")
    else:
        if not os.path.isfile(in_name + "0.mtx"):
            print(f"FAILED TO FIND PART FILES FOR {out_name} ; SKIPPING")
        else:
            cmd = ["./res", in_name, size, out_name]
            status = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
            if (status.returncode != 0): print(f"COMMAND {cmd} FAILED {status.returncode}\n{status.stdout}")

for folder in folders:
    for size in sizes:
        for ty in types:
            fname = f"{root}/{folder}/ModifiedParallel{size}by{size}{ty}_part_"
            out_fname = f"{output_root}/{folder}/MP{size}{ty}.mtx"
            resurrect_file(fname, out_fname, size)
            fname = f"{root}/{folder}/ClassicParallel{size}by{size}{ty}_part_"
            out_fname = f"{output_root}/{folder}/CP{size}{ty}.mtx"
            resurrect_file(fname, out_fname, size)

for folder in folders:
    cmd = ["./ortho", "-r", f"{output_root}/{folder}/"]
    status = subprocess.run(cmd)
