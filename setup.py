#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note: To use the 'upload' functionality of this file, you must:
#   $ pip install twine

import io
import os
import sys
import re
import shutil
from shutil import rmtree
import textwrap
import shlex
import subprocess

from setuptools import find_packages, setup, Command, Extension
from setuptools.command.build_ext import build_ext
from distutils.errors import CompileError, DistutilsError, DistutilsPlatformError, LinkError, DistutilsSetupError
from distutils import log as distutils_logger
from distutils.version import LooseVersion
import traceback

import pre_setup

server_lib = Extension('byteps.server.c_lib', [])
tensorflow_lib = Extension('byteps.tensorflow.c_lib', [])
mxnet_lib = Extension('byteps.mxnet.c_lib', [])
pytorch_lib = Extension('byteps.torch.c_lib', [])

# Package meta-data.
NAME = 'byteps'
DESCRIPTION = 'A high-performance cross-framework Parameter Server for Deep Learning'
URL = 'https://github.com/bytedance/byteps'
EMAIL = 'lab-hr@bytedance.com'
AUTHOR = 'Bytedance Inc.'
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
        os.system(
            '{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

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
    default_flags = ['-std=c++11', '-fPIC', '-Ofast', '-Wall', '-fopenmp', '-march=native']
    flags_to_try = []
    if sys.platform == 'darwin':
        # Darwin most likely will have Clang, which has libc++.
        flags_to_try = [default_flags + ['-stdlib=libc++'],
                        default_flags]
    else:
        flags_to_try = [default_flags ,
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
    ld_flags = ['-Wl,--version-script=byteps.lds', '-fopenmp']
    flags_to_try = []
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

def has_rdma_header():
    ret_code = subprocess.call(
        "echo '#include <rdma/rdma_cma.h>' | cpp -H -o /dev/null 2>/dev/null", shell=True)
    if ret_code != 0:
        import warnings
        warnings.warn("\n\n No RDMA header file detected. Will disable RDMA for compilation! \n\n")
    return ret_code==0

def get_common_options(build_ext):
    cpp_flags = get_cpp_flags(build_ext)
    link_flags = get_link_flags(build_ext)

    MACROS = [('EIGEN_MPL2_ONLY', 1)]
    INCLUDES = ['3rdparty/ps-lite/include']
    SOURCES = ['byteps/common/common.cc',
               'byteps/common/operations.cc',
               'byteps/common/core_loops.cc',
               'byteps/common/global.cc',
               'byteps/common/logging.cc',
               'byteps/common/communicator.cc',
               'byteps/common/scheduled_queue.cc',
               'byteps/common/ready_table.cc',
               'byteps/common/shared_memory.cc',
               'byteps/common/nccl_manager.cc',
               'byteps/common/cpu_reducer.cc'] + [
               'byteps/common/compressor/compressor_registry.cc',
               'byteps/common/compressor/error_feedback.cc',
               'byteps/common/compressor/momentum.cc',
               'byteps/common/compressor/impl/dithering.cc',
               'byteps/common/compressor/impl/onebit.cc',
               'byteps/common/compressor/impl/randomk.cc',
               'byteps/common/compressor/impl/topk.cc',
               'byteps/common/compressor/impl/vanilla_error_feedback.cc',
               'byteps/common/compressor/impl/nesterov_momentum.cc']
    if "BYTEPS_USE_MPI" in os.environ and os.environ["BYTEPS_USE_MPI"] == "1":
        mpi_flags = get_mpi_flags()
        COMPILE_FLAGS = cpp_flags + \
            shlex.split(mpi_flags) + ["-DBYTEPS_USE_MPI"]
        LINK_FLAGS = link_flags + shlex.split(mpi_flags)
    else:
        COMPILE_FLAGS = cpp_flags
        LINK_FLAGS = link_flags
    
    LIBRARY_DIRS = []
    LIBRARIES = []

    nccl_include_dirs, nccl_lib_dirs, nccl_libs = get_nccl_vals()
    INCLUDES += nccl_include_dirs
    LIBRARY_DIRS += nccl_lib_dirs
    LIBRARIES += nccl_libs

    # RDMA and NUMA libs
    LIBRARIES += ['numa']

    # auto-detect rdma
    if has_rdma_header():
        LIBRARIES += ['rdmacm', 'ibverbs', 'rt']

    # ps-lite
    EXTRA_OBJECTS = ['3rdparty/ps-lite/build/libps.a',
                     '3rdparty/ps-lite/deps/lib/libzmq.a']

    return dict(MACROS=MACROS,
                INCLUDES=INCLUDES,
                SOURCES=SOURCES,
                COMPILE_FLAGS=COMPILE_FLAGS,
                LINK_FLAGS=LINK_FLAGS,
                LIBRARY_DIRS=LIBRARY_DIRS,
                LIBRARIES=LIBRARIES,
                EXTRA_OBJECTS=EXTRA_OBJECTS)


def build_server(build_ext, options):
    server_lib.define_macros = options['MACROS']
    server_lib.include_dirs = options['INCLUDES']
    server_lib.sources = ['byteps/server/server.cc',
                          'byteps/common/cpu_reducer.cc',
                          'byteps/common/logging.cc',
                          'byteps/common/common.cc'] + [
                          'byteps/common/compressor/compressor_registry.cc',
                          'byteps/common/compressor/error_feedback.cc',
                          'byteps/common/compressor/impl/dithering.cc',
                          'byteps/common/compressor/impl/onebit.cc',
                          'byteps/common/compressor/impl/randomk.cc',
                          'byteps/common/compressor/impl/topk.cc',
                          'byteps/common/compressor/impl/vanilla_error_feedback.cc']
    server_lib.extra_compile_args = options['COMPILE_FLAGS'] + \
        ['-DBYTEPS_BUILDING_SERVER']
    server_lib.extra_link_args = options['LINK_FLAGS']
    server_lib.extra_objects = options['EXTRA_OBJECTS']
    server_lib.library_dirs = options['LIBRARY_DIRS']

    # auto-detect rdma
    if has_rdma_header():
        server_lib.libraries = ['rdmacm', 'ibverbs', 'rt']
    else:
        server_lib.libraries = []

    build_ext.build_extension(server_lib)


def check_tf_version():
    try:
        import tensorflow as tf
        if LooseVersion(tf.__version__) < LooseVersion('1.1.0'):
            raise DistutilsPlatformError(
                'Your TensorFlow version %s is outdated.  '
                'BytePS requires tensorflow>=1.1.0' % tf.__version__)
    except ImportError:
        raise DistutilsPlatformError(
            'import tensorflow failed, is it installed?\n\n%s' % traceback.format_exc())
    except AttributeError:
        # This means that tf.__version__ was not exposed, which makes it *REALLY* old.
        raise DistutilsPlatformError(
            'Your TensorFlow version is outdated. BytePS requires tensorflow>=1.1.0')


def get_tf_include_dirs():
    import tensorflow as tf
    tf_inc = tf.sysconfig.get_include()
    return [tf_inc, '%s/external/nsync/public' % tf_inc]


def get_tf_lib_dirs():
    import tensorflow as tf
    tf_lib = tf.sysconfig.get_lib()
    return [tf_lib]


def get_tf_libs(build_ext, lib_dirs, cpp_flags):
    last_err = None
    for tf_libs in [['tensorflow_framework'], []]:
        try:
            lib_file = test_compile(build_ext, 'test_tensorflow_libs',
                                    library_dirs=lib_dirs, libraries=tf_libs,
                                    extra_compile_preargs=cpp_flags,
                                    code=textwrap.dedent('''\
                    void test() {
                    }
                    '''))

            from tensorflow.python.framework import load_library
            load_library.load_op_library(lib_file)

            return tf_libs
        except (CompileError, LinkError):
            last_err = 'Unable to determine -l link flags to use with TensorFlow (see error above).'
        except Exception:
            last_err = 'Unable to determine -l link flags to use with TensorFlow.  ' \
                       'Last error:\n\n%s' % traceback.format_exc()

    raise DistutilsPlatformError(last_err)


def get_tf_abi(build_ext, include_dirs, lib_dirs, libs, cpp_flags):
    last_err = None
    cxx11_abi_macro = '_GLIBCXX_USE_CXX11_ABI'
    for cxx11_abi in ['0', '1']:
        try:
            lib_file = test_compile(build_ext, 'test_tensorflow_abi',
                                    macros=[(cxx11_abi_macro, cxx11_abi)],
                                    include_dirs=include_dirs, library_dirs=lib_dirs,
                                    libraries=libs, extra_compile_preargs=cpp_flags,
                                    code=textwrap.dedent('''\
                #include <string>
                #include "tensorflow/core/framework/op.h"
                #include "tensorflow/core/framework/op_kernel.h"
                #include "tensorflow/core/framework/shape_inference.h"
                void test() {
                    auto ignore = tensorflow::strings::StrCat("a", "b");
                }
                '''))

            from tensorflow.python.framework import load_library
            load_library.load_op_library(lib_file)

            return cxx11_abi_macro, cxx11_abi
        except (CompileError, LinkError):
            last_err = 'Unable to determine CXX11 ABI to use with TensorFlow (see error above).'
        except Exception:
            last_err = 'Unable to determine CXX11 ABI to use with TensorFlow.  ' \
                       'Last error:\n\n%s' % traceback.format_exc()

    raise DistutilsPlatformError(last_err)


def get_tf_flags(build_ext, cpp_flags):
    import tensorflow as tf
    try:
        return tf.sysconfig.get_compile_flags(), tf.sysconfig.get_link_flags()
    except AttributeError:
        # fallback to the previous logic
        tf_include_dirs = get_tf_include_dirs()
        tf_lib_dirs = get_tf_lib_dirs()
        tf_libs = get_tf_libs(build_ext, tf_lib_dirs, cpp_flags)
        tf_abi = get_tf_abi(build_ext, tf_include_dirs,
                            tf_lib_dirs, tf_libs, cpp_flags)

        compile_flags = []
        for include_dir in tf_include_dirs:
            compile_flags.append('-I%s' % include_dir)
        if tf_abi:
            compile_flags.append('-D%s=%s' % tf_abi)

        link_flags = []
        for lib_dir in tf_lib_dirs:
            link_flags.append('-L%s' % lib_dir)
        for lib in tf_libs:
            link_flags.append('-l%s' % lib)

        return compile_flags, link_flags


def build_tf_extension(build_ext, options):
    check_tf_version()
    tf_compile_flags, tf_link_flags = get_tf_flags(
        build_ext, options['COMPILE_FLAGS'])

    # We assume we have CUDA
    cuda_include_dirs, cuda_lib_dirs = get_cuda_dirs(
        build_ext, options['COMPILE_FLAGS'])
    options['MACROS'] += [('HAVE_CUDA', '1')]
    options['INCLUDES'] += cuda_include_dirs
    options['LIBRARY_DIRS'] += cuda_lib_dirs
    options['LIBRARIES'] += ['cudart']

    tensorflow_lib.define_macros = options['MACROS']
    tensorflow_lib.include_dirs = options['INCLUDES']
    tensorflow_lib.sources = options['SOURCES'] + \
        ['byteps/tensorflow/ops.cc']
    tensorflow_lib.extra_compile_args = options['COMPILE_FLAGS'] + \
        tf_compile_flags
    tensorflow_lib.extra_link_args = options['LINK_FLAGS'] + tf_link_flags
    tensorflow_lib.library_dirs = options['LIBRARY_DIRS']
    tensorflow_lib.libraries = options['LIBRARIES']
    tensorflow_lib.extra_objects = options['EXTRA_OBJECTS']

    build_ext.build_extension(tensorflow_lib)


def check_mx_version():
    try:
        import mxnet as mx
        if mx.__version__ < '1.4.0':
            raise DistutilsPlatformError(
                'Your MXNet version %s is outdated.  '
                'BytePS requires mxnet>=1.4.0' % mx.__version__)
    except ImportError:
        raise DistutilsPlatformError(
            'import mxnet failed, is it installed?\n\n%s' % traceback.format_exc())
    except AttributeError:
        raise DistutilsPlatformError(
            'Your MXNet version is outdated. BytePS requires mxnet>=1.4.0')


def get_mx_include_dirs():
    try:
        import mxnet as mx
        path = mx.libinfo.find_include_path()
        return path
    except:
        # Try to find the path automatically
        tmp_mxnet_dir = os.getenv(
            "BYTEPS_SERVER_MXNET_PATH", "/root/mxnet15-rdma")
        MXNET_ROOT = os.getenv("MXNET_SOURCE_ROOT", tmp_mxnet_dir)
        return os.path.join(MXNET_ROOT, 'include/')


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
            return mx_libs
        except (CompileError, LinkError):
            last_err = 'Unable to determine -l link flags to use with MXNet (see error above).'
        except Exception:
            last_err = 'Unable to determine -l link flags to use with MXNet.  ' \
                       'Last error:\n\n%s' % traceback.format_exc()

    raise DistutilsPlatformError(last_err)


def get_mx_flags(build_ext, cpp_flags):
    mx_include_dirs = [get_mx_include_dirs()]
    mx_lib_dirs = get_mx_lib_dirs()
    mx_libs = get_mx_libs(build_ext, mx_lib_dirs, cpp_flags)

    compile_flags = []
    has_mkldnn = is_mx_mkldnn()
    for include_dir in mx_include_dirs:
        compile_flags.append('-I%s' % include_dir)
        if has_mkldnn:
            mkldnn_include = os.path.join(include_dir, 'mkldnn')
            compile_flags.append('-I%s' % mkldnn_include)

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

    cuda_home = os.environ.get('BYTEPS_CUDA_HOME')
    if cuda_home:
        cuda_include_dirs += ['%s/include' % cuda_home]
        cuda_lib_dirs += ['%s/lib' % cuda_home, '%s/lib64' % cuda_home]

    cuda_include = os.environ.get('BYTEPS_CUDA_INCLUDE')
    if cuda_include:
        cuda_include_dirs += [cuda_include]

    cuda_lib = os.environ.get('BYTEPS_CUDA_LIB')
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
            'Please specify correct CUDA location with the BYTEPS_CUDA_HOME '
            'environment variable or combination of BYTEPS_CUDA_INCLUDE and '
            'BYTEPS_CUDA_LIB environment variables.\n\n'
            'BYTEPS_CUDA_HOME - path where CUDA include and lib directories can be found\n'
            'BYTEPS_CUDA_INCLUDE - path to CUDA include directory\n'
            'BYTEPS_CUDA_LIB - path to CUDA lib directory')

    return cuda_include_dirs, cuda_lib_dirs


def get_nccl_vals():
    nccl_include_dirs = []
    nccl_lib_dirs = []
    nccl_libs = []

    nccl_home = os.environ.get('BYTEPS_NCCL_HOME', '/usr/local/nccl')
    if nccl_home:
        nccl_include_dirs += ['%s/include' % nccl_home]
        nccl_lib_dirs += ['%s/lib' % nccl_home, '%s/lib64' % nccl_home]

    nccl_link_mode = os.environ.get('BYTEPS_NCCL_LINK', 'STATIC')
    if nccl_link_mode.upper() == 'SHARED':
        nccl_libs += ['nccl']
    else:
        nccl_libs += ['nccl_static']

    return nccl_include_dirs, nccl_lib_dirs, nccl_libs

def is_mx_mkldnn():
    try:
        from mxnet import runtime
        features = runtime.Features()
        return features.is_enabled('MKLDNN')
    except Exception:
        msg = 'INFO: Cannot detect if MKLDNN is enabled in MXNet. Please \
            set MXNET_USE_MKLDNN=1 if MKLDNN is enabled in your MXNet build.'
        if 'linux' not in sys.platform:
            # MKLDNN is only enabled by default in MXNet Linux build. Return 
            # False by default for non-linux build but still allow users to 
            # enable it by using MXNET_USE_MKLDNN env variable. 
            print(msg)
            return os.environ.get('MXNET_USE_MKLDNN', '0') == '1'
        else:
            try:
                import mxnet as mx
                mx_libs = mx.libinfo.find_lib_path()
                for mx_lib in mx_libs:
                    output = subprocess.check_output(['readelf', '-d', mx_lib])
                    if 'mkldnn' in str(output):
                        return True
                return False
            except Exception:
                print(msg)
                return os.environ.get('MXNET_USE_MKLDNN', '0') == '1'


def build_mx_extension(build_ext, options):
    # clear ROLE -- installation does not need this
    os.environ.pop("DMLC_ROLE", None)

    check_mx_version()
    mx_compile_flags, mx_link_flags = get_mx_flags(
        build_ext, options['COMPILE_FLAGS'])

    mx_have_cuda = is_mx_cuda()
    macro_have_cuda = check_macro(options['MACROS'], 'HAVE_CUDA')
    if not mx_have_cuda and macro_have_cuda:
        raise DistutilsPlatformError(
            'BytePS build with GPU support was requested, but this MXNet '
            'installation does not support CUDA.')

    # Update HAVE_CUDA to mean that MXNet supports CUDA.
    if mx_have_cuda and not macro_have_cuda:
        cuda_include_dirs, cuda_lib_dirs = get_cuda_dirs(
            build_ext, options['COMPILE_FLAGS'])
        options['MACROS'] += [('HAVE_CUDA', '1')]
        options['INCLUDES'] += cuda_include_dirs
        options['LIBRARY_DIRS'] += cuda_lib_dirs
        options['LIBRARIES'] += ['cudart']

    mxnet_lib.define_macros = options['MACROS']
    if check_macro(options['MACROS'], 'HAVE_CUDA'):
        mxnet_lib.define_macros += [('MSHADOW_USE_CUDA', '1')]
    else:
        mxnet_lib.define_macros += [('MSHADOW_USE_CUDA', '0')]
    if is_mx_mkldnn():
        mxnet_lib.define_macros += [('MXNET_USE_MKLDNN', '1')]
    else:
        mxnet_lib.define_macros += [('MXNET_USE_MKLDNN', '0')]  
    mxnet_lib.define_macros += [('MSHADOW_USE_MKL', '0')]

    # use MXNet's DMLC headers first instead of ps-lite's
    options['INCLUDES'].insert(0, get_mx_include_dirs())
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
    mxnet_lib.extra_objects = options['EXTRA_OBJECTS']
    mxnet_lib.library_dirs = options['LIBRARY_DIRS']
    mxnet_lib.libraries = options['LIBRARIES']

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
                'BytePS requires torch>=1.0.1' % torch.__version__)
    except ImportError:
            print('import torch failed, is it installed?\n\n%s' %
                  traceback.format_exc())

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
    pytorch_compile_flags = ["-std=c++14" if flag == "-std=c++11" 
                             else flag for flag in options['COMPILE_FLAGS']]
    have_cuda = is_torch_cuda(build_ext, include_dirs=options['INCLUDES'],
                              extra_compile_args=pytorch_compile_flags)
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
                         extra_compile_args=pytorch_compile_flags,
                         extra_link_args=options['LINK_FLAGS'],
                         extra_objects=options['EXTRA_OBJECTS'],
                         library_dirs=options['LIBRARY_DIRS'],
                         libraries=options['LIBRARIES'])

    # Patch an existing pytorch_lib extension object.
    for k, v in ext.__dict__.items():
        pytorch_lib.__dict__[k] = v
    build_ext.build_extension(pytorch_lib)


# run the customize_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        pre_setup.setup()

        make_option = ""
        # To resolve tf-gcc incompatibility
        has_cxx_flag = False
        glibcxx_flag = False
        if not int(os.environ.get('BYTEPS_WITHOUT_TENSORFLOW', 0)):
            try:
                import tensorflow as tf
                make_option += 'ADD_CFLAGS="'
                for flag in tf.sysconfig.get_compile_flags():
                    if 'D_GLIBCXX_USE_CXX11_ABI' in flag:
                        has_cxx_flag = True
                        glibcxx_flag = False if (flag[-1]=='0') else True
                        make_option += flag + ' '
                        break
                make_option += '" '
            except:
                pass

        # To resolve torch-gcc incompatibility
        if not int(os.environ.get('BYTEPS_WITHOUT_PYTORCH', 0)):
            try:
                import torch
                torch_flag = torch.compiled_with_cxx11_abi()
                if has_cxx_flag:
                    if glibcxx_flag != torch_flag:
                        raise DistutilsError(
                            '-D_GLIBCXX_USE_CXX11_ABI is not consistent between TensorFlow and PyTorch, '
                            'consider install them separately.')
                    else:
                        pass
                else:
                    make_option += 'ADD_CFLAGS=-D_GLIBCXX_USE_CXX11_ABI=' + \
                                    str(int(torch_flag)) + ' '
                    has_cxx_flag = True
                    glibcxx_flag = torch_flag
            except:
                pass

        if not os.path.exists("3rdparty/ps-lite/build/libps.a") or \
           not os.path.exists("3rdparty/ps-lite/deps/lib"):
            if os.environ.get('CI', 'false') == 'false':
                make_option += "-j "
            if has_rdma_header():
                make_option += "USE_RDMA=1 "

            make_option += pre_setup.extra_make_option()


            make_process = subprocess.Popen('make ' + make_option,
                                            cwd='3rdparty/ps-lite',
                                            stdout=sys.stdout,
                                            stderr=sys.stderr,
                                            shell=True)
            make_process.communicate()
            if make_process.returncode:
                raise DistutilsSetupError('An ERROR occured while running the '
                                          'Makefile for the ps-lite library. '
                                          'Exit code: {0}'.format(make_process.returncode))

        options = get_common_options(self)
        if has_cxx_flag:
            options['COMPILE_FLAGS'] += ['-D_GLIBCXX_USE_CXX11_ABI=' + str(int(glibcxx_flag))]

        built_plugins = []
        try:
            build_server(self, options)
        except:
            raise DistutilsSetupError('An ERROR occured while building the server module.\n\n'
                                      '%s' % traceback.format_exc())

        # If PyTorch is installed, it must be imported before others, otherwise
        # we may get an error: dlopen: cannot load any more object with static TLS
        if not int(os.environ.get('BYTEPS_WITHOUT_PYTORCH', 0)):
            dummy_import_torch()

        if not int(os.environ.get('BYTEPS_WITHOUT_TENSORFLOW', 0)):
            try:
                build_tf_extension(self, options)
                built_plugins.append(True)
                print('INFO: Tensorflow extension is built successfully.')
            except:
                if not int(os.environ.get('BYTEPS_WITH_TENSORFLOW', 0)):
                    print('INFO: Unable to build TensorFlow plugin, will skip it.\n\n'
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
        if not int(os.environ.get('BYTEPS_WITHOUT_MXNET', 0)):
            # fix "libcuda.so.1 not found" issue
            cuda_home = os.environ.get('BYTEPS_CUDA_HOME', '/usr/local/cuda')
            cuda_stub_path = cuda_home + '/lib64/stubs'
            ln_command = "cd " + cuda_stub_path + "; ln -sf libcuda.so libcuda.so.1"
            os.system(ln_command)
            try:
                build_mx_extension(self, options)
                built_plugins.append(True)
                print('INFO: MXNet extension is built successfully.')
            except:
                if not int(os.environ.get('BYTEPS_WITH_MXNET', 0)):
                    print('INFO: Unable to build MXNet plugin, will skip it.\n\n'
                          '%s' % traceback.format_exc())
                    built_plugins.append(False)
                else:
                    raise
            finally:
                os.system("rm -rf " + cuda_stub_path + "/libcuda.so.1")

        if not built_plugins:
            print('INFO: Only server module is built.')
            return

        if not any(built_plugins):
            raise DistutilsError(
                'None of TensorFlow, MXNet, PyTorch plugins were built. See errors above.')


# Where the magic happens:

if os.path.exists('launcher/launch.py'):
    if not os.path.exists('bin'):
        os.mkdir('bin')
    shutil.copyfile('launcher/launch.py', 'bin/bpslaunch')

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
    license='Apache',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Operating System :: POSIX :: Linux'
    ],
    ext_modules=[server_lib, tensorflow_lib, mxnet_lib, pytorch_lib],
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
    scripts=['bin/bpslaunch']
)
