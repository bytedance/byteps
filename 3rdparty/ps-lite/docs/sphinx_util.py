import sys, os, subprocess


if not os.path.exists('../recommonmark'):
    subprocess.call('cd ..; git clone https://github.com/tqchen/recommonmark', shell = True)
else:
    subprocess.call('cd ../recommonmark; git pull', shell=True)

sys.path.insert(0, os.path.abspath('../recommonmark/'))

from recommonmark import parser, transform
MarkdownParser = parser.CommonMarkParser
AutoStructify = transform.AutoStructify

# MarkdownParser.github_doc_root = github_doc_root

def generate_doxygen_xml(app):
    """Run the doxygen make commands"""
    subprocess.call('doxygen')
