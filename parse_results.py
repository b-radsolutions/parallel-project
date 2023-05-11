
import re

HEADER = """Matrix size,Matrix type,Procedure Type,Frobenius Norm,Infinity Norm,One Norm"""

def parse_line(line: str):
    lp = line.split(", ")
    m = re.match("([MC]P)([\d]+)-([\d]+)([\w-]+).mtx",lp[-1])
    if m == None: 
        print("[ERROR]", line)
        return None
    groups = m.groups()
    if groups[1] != '32': return None
    name = "Modified"
    if groups[0] == 'CP': name = "Classic"
    return (groups[2], groups[3], name, lp[1], lp[2], lp[3])

def print_grouped():
    with open("results", "r") as file:
        data = file.read()
        lines = data.split("\n")[:-1]
        lines = lines[1::2]
        processed = {}
        for line in lines:
            line_data = parse_line(line)
            if line_data == None: continue
            key = line_data[0]
            if key in processed:
                processed[key][line_data[2]] = line_data[4]
            else:
                processed[key] = {line_data[2]: line_data[4]}
            # processed[line_data[0]].append(line_data)
        print("Matrix Size,Modified,Classical")
        for key in processed:
            row = processed[key]
            if not 'Classic' in row or not 'Modified' in row: continue 
            print(f"{key},{processed[key]['Modified']},{processed[key]['Classic']}")

def print_res():
    with open("results", "r") as file:
        data = file.read()
        lines = data.split("\n")[:-1]
        lines = lines[1::2]
        processed = []
        for line in lines:
            line_data = parse_line(line)
            if line_data == None: continue
            processed.append(line_data)
        print(HEADER)
        print("\n".join([",".join(row) for row in processed]))
        # Matrix size, type, procedure type, errors..., 

def rearrange_data(fname):
    with open(fname, "r") as file:
        lines = file.read().split("\n")
        print("\\begin{tabular}{c|c|c|c|c}")
        made_lines = []
        for line in lines:
            parts = line.split("    ")
            made_lines.append("&".join(parts))
            print
        print(" \\\\ ".join(made_lines))
        print("\\end{tabular}")

# rearrange_data("message.txt")
# print_res()
print_grouped()
