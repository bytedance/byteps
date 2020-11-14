#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note: To use the 'upload' functionality of this file, you must:
#   $ pip install twine

import io
import os
import sys
from shutil import rmtree
import textwrap

from setuptools import find_packages, setup, Command, Extension
from setuptools.command.build_ext import build_ext
from distutils.errors import CompileError, DistutilsError, DistutilsPlatformError, LinkError
import traceback


# Manually add MXNet include path
# TODO: find this path automatically
MXNET_ROOT = os.getenv("MXNET_ROOT", "/root/mxnet-incubator/")
os.environ["MXNET_INCLUDE_PATH"] = os.path.join(MXNET_ROOT, "include/")

# TODO: check MXNet config.mk or Makefile
os.environ["MXNET_USE_MKLDNN"] = "0"

mxnet_lib = Extension('bytescheduler.mxnet.c_lib', [])
pytorch_lib = Extension('bytescheduler.pytorch.c_lib', [])

# Package meta-data.
NAME = 'bytescheduler'
DESCRIPTION = 'Scheduler for distributed training'
URL = 'https://github.com/yhpeng-git/bytescheduler'
EMAIL = 'lab-hr@bytedance.com'
AUTHOR = 'ByteDance Inc.'
REQUIRES_PYTHON = '>=2.7.0'
VERSION = None

# What packages are required for this module to be executed?
REQUIRED = [
    'bayesian-optimization>=1.0.0',
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
    libtool_flags = ['-Wl,-exported_symbols_list,bytescheduler.exp']
    ld_flags = ['-Wl,--version-script=bytescheduler.lds']
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
    INCLUDES = []
    SOURCES = []
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
                'ByteScheduler requires mxnet>=1.4.0' % mx.__version__)
    except ImportError:
        raise DistutilsPlatformError(
            'import mxnet failed, is it installed?\n\n%s' % traceback.format_exc())
    except AttributeError:
        raise DistutilsPlatformError(
            'Your MXNet version is outdated. ByteScheduler requires mxnet>=1.4.0.')


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


def build_mx_extension(build_ext, options):
    check_mx_version()
    mx_compile_flags, mx_link_flags = get_mx_flags(
        build_ext, options['COMPILE_FLAGS'])

    mxnet_lib.define_macros = options['MACROS']
    if check_macro(options['MACROS'], 'HAVE_CUDA'):
        mxnet_lib.define_macros += [('MSHADOW_USE_CUDA', '1')]
    else:
        mxnet_lib.define_macros += [('MSHADOW_USE_CUDA', '0')]
    mxnet_lib.define_macros += [('MSHADOW_USE_MKL', '0')]
    mxnet_lib.include_dirs = options['INCLUDES']
    mxnet_lib.sources = options['SOURCES'] + \
        ['bytescheduler/mxnet/c_lib.cc']
    mxnet_lib.extra_compile_args = options['COMPILE_FLAGS'] + \
        mx_compile_flags
    mxnet_lib.extra_link_args = options['LINK_FLAGS'] + mx_link_flags
    mxnet_lib.library_dirs = options['LIBRARY_DIRS']
    mxnet_lib.libraries = options['LIBRARIES']

    # MKLDNN
    if "MXNET_USE_MKLDNN" in os.environ and int(os.environ["MXNET_USE_MKLDNN"]) > 0:
        raise DistutilsError(
            'ByteScheduler does not support MXNet with MKLDNN.')
        mxnet_lib.define_macros += [('MXNET_USE_MKLDNN', '1')]
        mxnet_lib.define_macros += [('USE_MKL', '1')]
        mxnet_lib.include_dirs += [os.path.join(MXNET_ROOT, '3rdparty/mkldnn/include/')]
        mxnet_lib.include_dirs += [os.path.join(MXNET_ROOT, 'src/operator/nn/mkldnn/')]
    build_ext.build_extension(mxnet_lib)


def check_torch_version():
    try:
        import torch
        if torch.__version__ < '1.0.0':
            raise DistutilsPlatformError(
                'Your torch version %s is outdated.  '
                'Horovod requires torch>=1.0.0' % torch.__version__)
        return torch.__version__
    except ImportError:
            print('import torch failed, is it installed?\n\n%s' % traceback.format_exc())


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
            'Bytescheduler build with GPU support was requested, but this PyTorch '
            'installation does not support CUDA.')

    # Update HAVE_CUDA to mean that PyTorch supports CUDA. Internally, we will be checking
    # HOROVOD_GPU_(ALLREDUCE|ALLGATHER|BROADCAST) to decide whether we should use GPU
    # version or transfer tensors to CPU memory for those operations.
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
                         sources=options['SOURCES'] + ['bytescheduler/pytorch/c_lib.cc',
                                                       'bytescheduler/pytorch/ready_event.cc',
                                                       'bytescheduler/pytorch/cuda_util.cc'],
                         extra_compile_args=options['COMPILE_FLAGS'],
                         extra_link_args=options['LINK_FLAGS'],
                         library_dirs=options['LIBRARY_DIRS'],
                         libraries=options['LIBRARIES'])

    # Patch an existing torch_mpi_lib_v2 extension object.
    for k, v in ext.__dict__.items():
        pytorch_lib.__dict__[k] = v
    build_ext.build_extension(pytorch_lib)


# run the customize_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        options = get_common_options(self)
        built_plugins = []
        if not int(os.environ.get('BYTESCHEDULER_WITHOUT_MXNET', 0)):
            try:
                build_mx_extension(self, options)
                built_plugins.append(True)
                print('INFO: MXNet extension is built successfully.')
            except:
                if not os.environ.get('BYTESCHEDULER_WITH_MXNET'):
                    print('INFO: Unable to build MXNet plugin, will skip it.\n\n'
                          '%s' % traceback.format_exc())
                    built_plugins.append(False)
                else:
                    raise
        if not int(os.environ.get('BYTESCHEDULER_WITHOUT_PYTORCH', 0)):
            try:
                torch_version = check_torch_version()
                build_torch_extension(self, options, torch_version)
                built_plugins.append(True)
                print('INFO: PyTorch extension is built successfully.')
            except:
                if not int(os.environ.get('BYTESCHEDULER_WITH_PYTORCH', 0)):
                    print('INFO: Unable to build PyTorch plugin, will skip it.\n\n'
                        '%s' % traceback.format_exc())
                    built_plugins.append(False)
                else:
                    raise

        if not any(built_plugins):
            print('None of MXNet, PyTorch plugins were built.')


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
