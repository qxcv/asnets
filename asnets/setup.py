import os

from distutils.ccompiler import new_compiler
from distutils.sysconfig import customize_compiler

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
from setuptools.command.install import install
from setuptools.dist import Distribution

# TF 1.14 probably won't work with custom extensions, pending TF bug
# #29643 (custom ops get wrong gradient)
TF_REQUIRE = 'tensorflow>=1.13.0,<1.14.0'


def _build_ext(ext_obj):
    """Build C/C++ implementation of ASNet-specific TF ops."""
    import tensorflow as tf
    compiler = new_compiler(compiler=None,
                            dry_run=ext_obj.dry_run,
                            force=ext_obj.force)
    customize_compiler(compiler)
    compiler.add_include_dir(tf.sysconfig.get_include())
    src = 'asnets/ops/_asnet_ops_impl.cc'
    dest = 'asnets/ops/_asnet_ops_impl.so'
    # Tests still pass with -Ofast, and it gives marginal speed improvement, so
    # I'm leaving it on. Might revisit later. Also revisit -march=native, which
    # will probably make distribution & Dockerisation a pain.
    extra_flags = ['-std=c++11', '-march=native', '-Ofast']
    objects = compiler.compile(
        [src],
        debug=True,
        extra_preargs=[*extra_flags, *tf.sysconfig.get_compile_flags()])
    compiler.link(compiler.SHARED_LIBRARY,
                  objects,
                  dest,
                  debug=True,
                  extra_postargs=[
                      *extra_flags, '-lstdc++', '-Wl,--no-undefined',
                      *tf.sysconfig.get_link_flags()
                  ])
    # cleanup: remove object files
    for obj in objects:
        os.unlink(obj)


class build_py_and_ext(build_py):
    # no need to handle planners; we install those on deploy
    def run(self):
        _build_ext(self)
        super().run()


class develop_and_planner_setup_and_ext(develop):
    # here we install planners AND build extensions
    def run(self):
        _build_ext(self)
        super().run()
        from asnets.fd_interface import try_install_fd
        try_install_fd()
        from asnets.ssipp_interface import try_install_ssipp_solver
        try_install_ssipp_solver()


class install_and_planner_setup(install):
    # no need to build extensions b/c that happens in build_py
    def run(self):
        super().run()
        from asnets.fd_interface import try_install_fd
        try_install_fd()
        from asnets.ssipp_interface import try_install_ssipp_solver
        try_install_ssipp_solver()


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        # create OS-specific wheels (per example TF op project on Github)
        return True


setup(
    name='asnets',
    version='0.0.1',
    packages=['asnets', 'experiments'],
    # putting this in setup_requires should ensure that we can import tf in
    # _build_ext during setup.py execution; putting it in install_requires
    # ensures that we also have tf at run time
    setup_requires=[TF_REQUIRE],
    install_requires=[
        # these are custom deps that can be a bit painful to install; if they
        # fail, try installing again just in case it was a transient error with
        # the compile scripts
        ('ssipp @ git+https://gitlab.com/qxcv/ssipp.git'
         '#sha1=49e922b4cbdf6d7278a34446ae6fc1732efb6fec'),
        ('mdpsim @ git+https://gitlab.cecs.anu.edu.au/u5568237/mdpsim.git'
         '#sha1=0881a3e86b15582b2e632904193c12d310bedda3'),

        # these are just vanilla PyPI deps & should be easy to install
        'rpyc>=4.0.2,<4.1.0',
        'tqdm>=4.14.0,<5.0',
        'joblib>=0.9.4',
        'numpy>=1.14,<1.17',
        'matplotlib>=3.0.2,<3.1.0',
        'Click>=7.0,<8.0',
        'crayons>=0.2.0,<0.3.0',
        'requests>=2.22.0,<3.0.0',
        'setproctitle>=1.1.10,<1.2.0',
        TF_REQUIRE,

        # ray (which benefits from psutil/setproctitle/boto3) is required for
        # run_experiments (and maybe other things in the future)
        'psutil>=5.6.3,<5.7.0',
        # 0.7.2 is required for Ray Tune, but breaks cluster (?)
        'ray>=0.7.1,<0.7.2',
        'boto3>=1.9.166,<1.10.0',
        'scikit-optimize>=0.5.2,<0.6.0',
        # ray tune needs pandas for some reason
        'pandas>=0.24.2,<0.25.0',

        # for the activation visualisation script (which is optional!), we also
        # need graph_tool, which can be installed manually by following the
        # instructions at https://graph-tool.skewed.de/
    ],
    # include_package_data=True, combined with our MANIFEST.in, ensures that
    # .so files are included
    include_package_data=True,
    distclass=BinaryDistribution,
    zip_safe=False,
    cmdclass={
        'develop': develop_and_planner_setup_and_ext,
        'install': install_and_planner_setup,
        'build_py': build_py_and_ext,
    })
