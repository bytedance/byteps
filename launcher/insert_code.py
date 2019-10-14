# python3 insert_code.py --target_file="tensor.py" --position="class Tensor(torch._C._TensorBase):" --content_file="insert_content.py"

import argparse
parser = argparse.ArgumentParser(description="Insert code to a file", 
							formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--target_file', type=str, default=None,
                    help='target file where you want to insert code')
parser.add_argument('--start', type=str, default=None,
                    help='the start position of where you want to insert code in the target file, must be a complete line.')
parser.add_argument('--end', type=str, default=None,
                    help='the end position of where you want to insert code in the target file,must be a complete line.')
parser.add_argument('--content_file', type=str, default=None,
                    help='A file containing contents that you want to insert, 0 indent level')
parser.add_argument('--content_str', type=str, default=None,
                    help='A file containing contents that you want to insert')
parser.add_argument('--indent_level', type=int, default=1,
                    help='A file containing contents that you want to insert')

args = parser.parse_args()

''' Get the target file content '''
with open(args.target_file, "r") as fp:
	target = fp.read()

''' Get the content we want to insert'''
if args.content_str:
	text = args.content_str
else:
	with open(args.content_file, "r") as fp:
		text = fp.read()

''' Search find the range where we want to insert content '''
start_pos = target.find(args.start) + len(args.start)
end_pos = target.find(args.end)
# print(target[(start_pos-len(args.start)):start_pos])
# print(target[end_pos:(end_pos+len(args.end))])

''' replace '\t' with four " " '''
text = text.replace("\t", " "*4)
''' uniform the indent level '''
INDENT = " "*(4 * args.indent_level)
text = text.replace("\n", "\n" + INDENT)

'''insert '''
if start_pos != -1 and end_pos != -1:
	target = target[:start_pos] \
			+ "\n\n" \
			+ INDENT + "## ---------------- automatic insert starts ---------------- \n" \
			+ INDENT + text + "\n" \
			+ INDENT + "## ---------------- automatic insert ends ---------------- \n\n" \
			+ target[end_pos:]

	with open(args.target_file, 'w') as fp:
		fp.write(target)