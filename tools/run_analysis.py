import subprocess
import os

root = "../../results/out"
folders = ['4', '8', '16', '32']
sizes = ['4', '16', '32', '64', '128', '256', '512', '1024', '2048', '4096', '8192']
types = ['dense', 'ill-conditioned', 'well-conditioned', 'sparse']
output_root = "../../results/reconstruct"

def resurrect_file(in_name, out_name, size):
    if os.path.isfile(out_name):
        # print(f"MATRIX FILE {out_name} ALREADY EXISTS ; SKIPPING")
        pass
    else:
        if not os.path.isfile(in_name + "0.mtx"):
            print(f"FAILED TO FIND PART FILES FOR {out_name} ; {(in_name + '0.mtx')} ; SKIPPING")
        else:
            cmd = ["./res", in_name, size, out_name]
            status = subprocess.run(cmd, stdout=subprocess.PIPE)
            if (status.returncode != 0): print(f"COMMAND {cmd} FAILED {status.returncode} ; {status.stdout}")

print("Reconstructing matrices:")
for folder in folders:
    for size in sizes:
        for ty in types:
            fname = f"{root}/{folder}/ModifiedParallel{size}by{size}{ty}_part_"
            out_fname = f"{output_root}/MP{folder}-{size}-{ty}.mtx"
            resurrect_file(fname, out_fname, folder)
            fname = f"{root}/{folder}/ClassicParallel{size}by{size}{ty}_part_"
            out_fname = f"{output_root}/CP{folder}-{size}-{ty}.mtx"
            resurrect_file(fname, out_fname, folder)

print("Analyzing matrices:")
cmd = ["./ortho", "-r", f"{output_root}/"]
status = subprocess.run(cmd)
