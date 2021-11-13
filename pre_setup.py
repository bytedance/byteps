from distutils.errors import CompileError, DistutilsError, DistutilsPlatformError, LinkError, DistutilsSetupError

# For internal use. Please do not modify this file.

def should_build_ucx():
    byteps_with_ucx = int(os.environ.get('BYTEPS_WITH_UCX', 0))
    has_prebuilt_ucx = os.environ.get('BYTEPS_UCX_HOME', '')
    return byteps_with_ucx and not has_prebuilt_ucx

def get_missing_deps(deps):
    missing_deps = []
    for item in deps:
        cmd = "dpkg -l {0} | grep -q ^ii".format(item)
        res = subprocess.run(cmd, shell=True)
        if res.returncode != 0:
            missing_deps.append(item)
    return missing_deps

def setup():
    return

def extra_make_option():
    return ""

# uncomment this line to specify the path to ucx.tar.gz
# must be a absolute path
# ucx_tarball_path = ""
