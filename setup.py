#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note: To use the 'upload' functionality of this file, you must:
#   $ pip install twine

import io
import os
import sys
import re
from shutil import rmtree
import textwrap
import shlex
import subprocess

from setuptools import find_packages, setup, Command, Extension
from setuptools.command.build_ext import build_ext
from distutils.errors import CompileError, DistutilsError, DistutilsPlatformError, LinkError
import traceback

# Not sure why the OS global env does not work
# Manually add MXNet include path
# TODO: find this path automatically
if os.path.exists("/root/mxnet-rdma"): 
    tmp_mxnet_dir = "/root/mxnet-rdma"
else:
    tmp_mxnet_dir = "/root/mxnet15-rdma"
MXNET_ROOT = os.getenv("MXNET_SOURCE_ROOT", tmp_mxnet_dir)
os.environ["MXNET_INCLUDE_PATH"] = os.path.join(MXNET_ROOT, "include/")

mxnet_lib = Extension('byteps.mxnet.c_lib', [])
pytorch_lib = Extension('byteps.torch.c_lib', [])

# Package meta-data.
NAME = 'byteps'
DESCRIPTION = 'A high-performance cross-framework Parameter Server for Deep Learning'
URL = 'https://code.byted.org/zhuyibo/byteps'
EMAIL = 'lab-hr@bytedance.com'
AUTHOR = 'ByteDance Inc.'
REQUIRES_PYTHON = '>=2.7.0'
VERSION = None

# What packages are required for this module to be executed?
REQUIRED = [
    # 'cffi>=1.4.0',
]

# What packages are optional?
EXTRAS = {
    # 'fancy feature': ['django'],
}

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except OSError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    with open(os.path.join(here, NAME, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION

def is_build_action():
    if len(sys.argv) <= 1:
        return False

    if sys.argv[1].startswith('build'):
        return True

    if sys.argv[1].startswith('bdist'):
        return True

    if sys.argv[1].startswith('install'):
        return True

class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status('Uploading the package to PyPI via Twine…')
        os.system('twine upload dist/*')

        self.status('Pushing git tags…')
        os.system('git tag v{0}'.format(about['__version__']))
        os.system('git push --tags')
        
        sys.exit()

# Start to build c libs
# ------------------------------------------------
def test_compile(build_ext, name, code, libraries=None, include_dirs=None, library_dirs=None,
                 macros=None, extra_compile_preargs=None, extra_link_preargs=None):
    test_compile_dir = os.path.join(build_ext.build_temp, 'test_compile')
    if not os.path.exists(test_compile_dir):
        os.makedirs(test_compile_dir)

    source_file = os.path.join(test_compile_dir, '%s.cc' % name)
    with open(source_file, 'w') as f:
        f.write(code)

    compiler = build_ext.compiler
    [object_file] = compiler.object_filenames([source_file])
    shared_object_file = compiler.shared_object_filename(
        name, output_dir=test_compile_dir)

    compiler.compile([source_file], extra_preargs=extra_compile_preargs,
                     include_dirs=include_dirs, macros=macros)
    compiler.link_shared_object(
        [object_file], shared_object_file, libraries=libraries, library_dirs=library_dirs,
        extra_preargs=extra_link_preargs)

    return shared_object_file

def get_mpi_flags():
    show_command = os.environ.get('BYTEPS_MPICXX_SHOW', 'mpicxx -show')
    try:
        mpi_show_output = subprocess.check_output(
            shlex.split(show_command), universal_newlines=True).strip()
        mpi_show_args = shlex.split(mpi_show_output)
        if not mpi_show_args[0].startswith('-'):
            # Open MPI and MPICH print compiler name as a first word, skip it
            mpi_show_args = mpi_show_args[1:]
        # strip off compiler call portion and always escape each arg
        return ' '.join(['"' + arg.replace('"', '"\'"\'"') + '"'
                         for arg in mpi_show_args])
    except Exception:
        raise DistutilsPlatformError(
            '%s failed (see error below), is MPI in $PATH?\n'
            'Note: If your version of MPI has a custom command to show compilation flags, '
            'please specify it with the BYTEPS_MPICXX_SHOW environment variable.\n\n'
            '%s' % (show_command, traceback.format_exc()))

def get_cpp_flags(build_ext):
    last_err = None
    default_flags = ['-std=c++11', '-fPIC', '-O2', '-Wall']
    avx_flags = ['-mf16c', '-mavx']
    if sys.platform == 'darwin':
        # Darwin most likely will have Clang, which has libc++.
        flags_to_try = [default_flags + ['-stdlib=libc++'] + avx_flags,
                        default_flags + avx_flags,
                        default_flags + ['-stdlib=libc++'],
                        default_flags]
    else:
        flags_to_try = [default_flags + avx_flags,
                        default_flags + ['-stdlib=libc++'] + avx_flags,
                        default_flags,
                        default_flags + ['-stdlib=libc++']]
    for cpp_flags in flags_to_try:
        try:
            test_compile(build_ext, 'test_cpp_flags', extra_compile_preargs=cpp_flags,
                         code=textwrap.dedent('''\
                    #include <unordered_map>
                    void test() {
                    }
                    '''))

            return cpp_flags
        except (CompileError, LinkError):
            last_err = 'Unable to determine C++ compilation flags (see error above).'
        except Exception:
            last_err = 'Unable to determine C++ compilation flags.  ' \
                       'Last error:\n\n%s' % traceback.format_exc()

    raise DistutilsPlatformError(last_err)

def get_link_flags(build_ext):
    last_err = None
    libtool_flags = ['-Wl,-exported_symbols_list,byteps.exp']
    ld_flags = ['-Wl,--version-script=byteps.lds']
    if sys.platform == 'darwin':
        flags_to_try = [libtool_flags, ld_flags]
    else:
        flags_to_try = [ld_flags, libtool_flags]
    for link_flags in flags_to_try:
        try:
            test_compile(build_ext, 'test_link_flags', extra_link_preargs=link_flags,
                         code=textwrap.dedent('''\
                    void test() {
                    }
                    '''))

            return link_flags
        except (CompileError, LinkError):
            last_err = 'Unable to determine C++ link flags (see error above).'
        except Exception:
            last_err = 'Unable to determine C++ link flags.  ' \
                       'Last error:\n\n%s' % traceback.format_exc()

    raise DistutilsPlatformError(last_err)


def get_common_options(build_ext):
    cpp_flags = get_cpp_flags(build_ext)
    link_flags = get_link_flags(build_ext)
    
    MACROS = [('EIGEN_MPL2_ONLY', 1)]
    INCLUDES = ['3rdparty/ps-lite/include']
    SOURCES = ['byteps/common/common.cc',
               'byteps/common/operations.cc',
               'byteps/common/global.cc',
               'byteps/common/logging.cc',
               'byteps/common/communicator.cc',
               'byteps/common/scheduled_queue.cc']
    if "BYTEPS_USE_MPI" in os.environ and os.environ["BYTEPS_USE_MPI"] == "1":
        mpi_flags = get_mpi_flags()
        COMPILE_FLAGS = cpp_flags + shlex.split(mpi_flags) + ["-DBYTEPS_USE_MPI"]
        LINK_FLAGS = link_flags + shlex.split(mpi_flags)
    else:
        COMPILE_FLAGS = cpp_flags
        LINK_FLAGS = link_flags
    LIBRARY_DIRS = []
    LIBRARIES = []

    return dict(MACROS=MACROS,
                INCLUDES=INCLUDES,
                SOURCES=SOURCES,
                COMPILE_FLAGS=COMPILE_FLAGS,
                LINK_FLAGS=LINK_FLAGS,
                LIBRARY_DIRS=LIBRARY_DIRS,
                LIBRARIES=LIBRARIES)


def check_mx_version():
    try:
        import mxnet as mx
        if mx.__version__ < '1.4.0':
            raise DistutilsPlatformError(
                'Your MXNet version %s is outdated.  '
                'byteps requires mxnet>=1.4.0' % mx.__version__)
    except ImportError:
        raise DistutilsPlatformError(
            'import mxnet failed, is it installed?\n\n%s' % traceback.format_exc())
    except AttributeError:
        raise DistutilsPlatformError(
            'Your MXNet version is outdated. byteps requires mxnet>1.3.0')

def get_mx_include_dirs():
    import mxnet as mx
    if mx.__version__ < '1.4.0':
        return os.path.join(MXNET_ROOT, 'include/')
    else:
        return [mx.libinfo.find_include_path()]


def get_mx_lib_dirs():
    import mxnet as mx
    mx_libs = mx.libinfo.find_lib_path()
    mx_lib_dirs = [os.path.dirname(mx_lib) for mx_lib in mx_libs]
    return mx_lib_dirs


def get_mx_libs(build_ext, lib_dirs, cpp_flags):
    last_err = None
    cpp_flags.append('-DDMLC_USE_RDMA')
    for mx_libs in [['mxnet'], []]:
        try:
            lib_file = test_compile(build_ext, 'test_mx_libs',
                                    library_dirs=lib_dirs, libraries=mx_libs,
                                    extra_compile_preargs=cpp_flags,
                                    code=textwrap.dedent('''\
                    void test() {
                    }
                    '''))
            mx_libs.append('rdmacm')
            mx_libs.append('ibverbs')
            return mx_libs
        except (CompileError, LinkError):
            last_err = 'Unable to determine -l link flags to use with MXNet (see error above).'
        except Exception:
            last_err = 'Unable to determine -l link flags to use with MXNet.  ' \
                       'Last error:\n\n%s' % traceback.format_exc()

    raise DistutilsPlatformError(last_err)

def get_mx_flags(build_ext, cpp_flags):
    mx_include_dirs = get_mx_include_dirs()
    mx_lib_dirs = get_mx_lib_dirs()
    mx_libs = get_mx_libs(build_ext, mx_lib_dirs, cpp_flags)

    compile_flags = []
    for include_dir in mx_include_dirs:
        compile_flags.append('-I%s' % include_dir)

    link_flags = []
    for lib_dir in mx_lib_dirs:
        link_flags.append('-Wl,-rpath,%s' % lib_dir)
        link_flags.append('-L%s' % lib_dir)

    for lib in mx_libs:
        link_flags.append('-l%s' % lib)

    return compile_flags, link_flags

def check_macro(macros, key):
    return any(k == key and v for k, v in macros)

def set_macro(macros, key, new_value):
    if any(k == key for k, _ in macros):
        return [(k, new_value if k == key else v) for k, v in macros]
    else:
        return macros + [(key, new_value)]

def is_mx_cuda():
    try:
        from mxnet import runtime
        features = runtime.Features()
        return features.is_enabled('CUDA')
    except Exception:
        if 'linux' in sys.platform:
            try:
                import mxnet as mx
                mx_libs = mx.libinfo.find_lib_path()
                for mx_lib in mx_libs:
                    output = subprocess.check_output(['readelf', '-d', mx_lib])
                    if 'cuda' in str(output):
                        return True
                return False
            except Exception:
                return False
    return False

def get_cuda_dirs(build_ext, cpp_flags):
    cuda_include_dirs = []
    cuda_lib_dirs = []

    cuda_home = os.environ.get('HOROVOD_CUDA_HOME')
    if cuda_home:
        cuda_include_dirs += ['%s/include' % cuda_home]
        cuda_lib_dirs += ['%s/lib' % cuda_home, '%s/lib64' % cuda_home]

    cuda_include = os.environ.get('HOROVOD_CUDA_INCLUDE')
    if cuda_include:
        cuda_include_dirs += [cuda_include]

    cuda_lib = os.environ.get('HOROVOD_CUDA_LIB')
    if cuda_lib:
        cuda_lib_dirs += [cuda_lib]

    if not cuda_include_dirs and not cuda_lib_dirs:
        # default to /usr/local/cuda
        cuda_include_dirs += ['/usr/local/cuda/include']
        cuda_lib_dirs += ['/usr/local/cuda/lib', '/usr/local/cuda/lib64']

    try:
        test_compile(build_ext, 'test_cuda', libraries=['cudart'], include_dirs=cuda_include_dirs,
                     library_dirs=cuda_lib_dirs, extra_compile_preargs=cpp_flags,
                     code=textwrap.dedent('''\
            #include <cuda_runtime.h>
            void test() {
                cudaSetDevice(0);
            }
            '''))
    except (CompileError, LinkError):
        raise DistutilsPlatformError(
            'CUDA library was not found (see error above).\n'
            'Please specify correct CUDA location with the HOROVOD_CUDA_HOME '
            'environment variable or combination of HOROVOD_CUDA_INCLUDE and '
            'HOROVOD_CUDA_LIB environment variables.\n\n'
            'HOROVOD_CUDA_HOME - path where CUDA include and lib directories can be found\n'
            'HOROVOD_CUDA_INCLUDE - path to CUDA include directory\n'
            'HOROVOD_CUDA_LIB - path to CUDA lib directory')

    return cuda_include_dirs, cuda_lib_dirs

def build_mx_extension(build_ext, options):
    check_mx_version()
    mx_compile_flags, mx_link_flags = get_mx_flags(
        build_ext, options['COMPILE_FLAGS'])

    mx_have_cuda = is_mx_cuda()
    macro_have_cuda = check_macro(options['MACROS'], 'HAVE_CUDA')
    if not mx_have_cuda and macro_have_cuda:
        raise DistutilsPlatformError(
            'Horovod build with GPU support was requested, but this MXNet '
            'installation does not support CUDA.')

    # Update HAVE_CUDA to mean that MXNet supports CUDA. Internally, we will be checking
    # HOROVOD_GPU_(ALLREDUCE|ALLGATHER|BROADCAST) to decide whether we should use GPU
    # version or transfer tensors to CPU memory for those operations.
    if mx_have_cuda and not macro_have_cuda:
        cuda_include_dirs, cuda_lib_dirs = get_cuda_dirs(build_ext, options['COMPILE_FLAGS'])
        options['MACROS'] += [('HAVE_CUDA', '1')]
        options['INCLUDES'] += cuda_include_dirs
        options['LIBRARY_DIRS'] += cuda_lib_dirs
        options['LIBRARIES'] += ['cudart']

    mxnet_lib.define_macros = options['MACROS']
    if check_macro(options['MACROS'], 'HAVE_CUDA'):
        mxnet_lib.define_macros += [('MSHADOW_USE_CUDA', '1')]
    else:
        mxnet_lib.define_macros += [('MSHADOW_USE_CUDA', '0')]
    mxnet_lib.define_macros += [('MSHADOW_USE_MKL', '0')]

    # use MXNet's DMLC headers first instead of ps-lite's
    options['INCLUDES'].insert(0, os.environ["MXNET_INCLUDE_PATH"])
    mxnet_lib.include_dirs = options['INCLUDES']
    mxnet_lib.sources = options['SOURCES'] + \
        ['byteps/mxnet/ops.cc',
         'byteps/mxnet/ready_event.cc',
         'byteps/mxnet/tensor_util.cc',
         'byteps/mxnet/cuda_util.cc',
         'byteps/mxnet/adapter.cc']
    mxnet_lib.extra_compile_args = options['COMPILE_FLAGS'] + \
        mx_compile_flags
    mxnet_lib.extra_link_args = options['LINK_FLAGS'] + mx_link_flags
    mxnet_lib.extra_objects = ['3rdparty/lib/libps.a']
    mxnet_lib.library_dirs = options['LIBRARY_DIRS']
    mxnet_lib.libraries = options['LIBRARIES']

    # build mxnet from source code to have these headers
    # mxnet_lib.include_dirs += [os.path.join(MXNET_ROOT, 'include/'), os.path.join(MXNET_ROOT, '3rdparty/ps-lite/include/')]
    build_ext.build_extension(mxnet_lib)

def dummy_import_torch():
    try:
        import torch
    except:
        pass

def parse_version(version_str):
    if "dev" in version_str:
        return 9999999999
    m = re.match('^(\d+)(?:\.(\d+))?(?:\.(\d+))?(?:\.(\d+))?', version_str)
    if m is None:
        return None

    # turn version string to long integer
    version = int(m.group(1)) * 10 ** 9
    if m.group(2) is not None:
        version += int(m.group(2)) * 10 ** 6
    if m.group(3) is not None:
        version += int(m.group(3)) * 10 ** 3
    if m.group(4) is not None:
        version += int(m.group(4))
    return version

def check_torch_version():
    try:
        import torch
        if torch.__version__ < '1.0.1':
            raise DistutilsPlatformError(
                'Your torch version %s is outdated.  '
                'Horovod requires torch>=1.0.1' % torch.__version__)
    except ImportError:
            print('import torch failed, is it installed?\n\n%s' % traceback.format_exc())

    # parse version
    version = parse_version(torch.__version__)
    if version is None:
        raise DistutilsPlatformError(
            'Unable to determine PyTorch version from the version string \'%s\'' % torch.__version__)
    return version

def is_torch_cuda(build_ext, include_dirs, extra_compile_args):
    try:
        from torch.utils.cpp_extension import include_paths
        test_compile(build_ext, 'test_torch_cuda', include_dirs=include_dirs + include_paths(cuda=True),
                     extra_compile_preargs=extra_compile_args, code=textwrap.dedent('''\
            #include <THC/THC.h>
            void test() {
            }
            '''))
        return True
    except (CompileError, LinkError, EnvironmentError):
        print('INFO: Above error indicates that this PyTorch installation does not support CUDA.')
        return False

def build_torch_extension(build_ext, options, torch_version):
    have_cuda = is_torch_cuda(build_ext, include_dirs=options['INCLUDES'],
                                 extra_compile_args=options['COMPILE_FLAGS'])
    if not have_cuda and check_macro(options['MACROS'], 'HAVE_CUDA'):
        raise DistutilsPlatformError(
            'byteps build with GPU support was requested, but this PyTorch '
            'installation does not support CUDA.')

    # Update HAVE_CUDA to mean that PyTorch supports CUDA.
    updated_macros = set_macro(
        options['MACROS'], 'HAVE_CUDA', str(int(have_cuda)))

    # Export TORCH_VERSION equal to our representation of torch.__version__. Internally it's
    # used for backwards compatibility checks.
    updated_macros = set_macro(
       updated_macros, 'TORCH_VERSION', str(torch_version))

    # Always set _GLIBCXX_USE_CXX11_ABI, since PyTorch can only detect whether it was set to 1.
    import torch
    updated_macros = set_macro(updated_macros, '_GLIBCXX_USE_CXX11_ABI',
                               str(int(torch.compiled_with_cxx11_abi())))

    # PyTorch requires -DTORCH_API_INCLUDE_EXTENSION_H
    updated_macros = set_macro(
        updated_macros, 'TORCH_API_INCLUDE_EXTENSION_H', '1')

    if have_cuda:
        from torch.utils.cpp_extension import CUDAExtension as TorchExtension
    else:
        # CUDAExtension fails with `ld: library not found for -lcudart` if CUDA is not present
        from torch.utils.cpp_extension import CppExtension as TorchExtension
    ext = TorchExtension(pytorch_lib.name,
                         define_macros=updated_macros,
                         include_dirs=options['INCLUDES'],
                         sources=options['SOURCES'] + ['byteps/torch/ops.cc',
                                                       'byteps/torch/ready_event.cc',
                                                       'byteps/torch/cuda_util.cc',
                                                       'byteps/torch/adapter.cc',
                                                       'byteps/torch/handle_manager.cc'],
                         extra_compile_args=options['COMPILE_FLAGS'],
                         extra_link_args=options['LINK_FLAGS'] + ['-Wl,-rpath,3rdparty/ps-lite/deps/lib',
                                                                  '-L3rdparty/ps-lite/deps/lib',
                                                                  '-lrdmacm',
                                                                  '-libverbs',
                                                                  '-lprotobuf',
                                                                  '-lzmq'],
                         extra_objects = ['3rdparty/ps-lite/build/libps.a'],
                         library_dirs=options['LIBRARY_DIRS'],
                         libraries=options['LIBRARIES'])

    # Patch an existing pytorch_lib extension object.
    for k, v in ext.__dict__.items():
        pytorch_lib.__dict__[k] = v
    build_ext.build_extension(pytorch_lib)

# run the customize_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        options = get_common_options(self)
        built_plugins = []

        # If PyTorch is installed, it must be imported before others, otherwise
        # we may get an error: dlopen: cannot load any more object with static TLS
        if not os.environ.get('BYTEPS_WITHOUT_PYTORCH'):
            dummy_import_torch()

        if not int(os.environ.get('BYTEPS_WITHOUT_MXNET', 0)):
            try:
                build_mx_extension(self, options)
                built_plugins.append(True)
                print('INFO: MXNet extension is built successfully.')
            except:
                if not os.environ.get('BYTEPS_WITH_MXNET'):
                    print('INFO: Unable to build MXNet plugin, will skip it.\n\n'
                          '%s' % traceback.format_exc())
                    built_plugins.append(False)
                else:
                    raise
        if not int(os.environ.get('BYTEPS_WITHOUT_PYTORCH', 0)):
            try:
                torch_version = check_torch_version()
                build_torch_extension(self, options, torch_version)
                built_plugins.append(True)
                print('INFO: PyTorch extension is built successfully.')
            except:
                if not int(os.environ.get('BYTEPS_WITH_PYTORCH', 0)):
                    print('INFO: Unable to build PyTorch plugin, will skip it.\n\n'
                        '%s' % traceback.format_exc())
                    built_plugins.append(False)
                else:
                    raise

        if not built_plugins:
            raise DistutilsError(
                'MXNet, PyTorch plugins were excluded from build. Aborting.')
        if not any(built_plugins):
            raise DistutilsError(
                'None of MXNet, PyTorch plugins were built. See errors above.')

# Where the magic happens:
setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=('tests',)),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license='MIT',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
    ext_modules=[mxnet_lib, pytorch_lib],
    # $ setup.py publish support.
    cmdclass={
        'upload': UploadCommand,
        'build_ext': custom_build_ext
    },
    # cffi is required for PyTorch
    # If cffi is specified in setup_requires, it will need libffi to be installed on the machine,
    # which is undesirable.  Luckily, `install` action will install cffi before executing build,
    # so it's only necessary for `build*` or `bdist*` actions.
    setup_requires=[],
)
