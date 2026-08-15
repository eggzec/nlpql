py_mod = import('python')
py = py_mod.find_installation(pure: false)
py_dep = py.dependency()

f2py = find_program('f2py', required: true)

incdir_numpy = run_command(py,
  ['-c', 'import numpy; print(numpy.get_include())'],
  check: true
).stdout().strip()

incdir_f2py = run_command(py,
  ['-c', 'import numpy.f2py; print(numpy.f2py.get_include())'],
  check: true
).stdout().strip()

fortranobject_c = incdir_f2py / 'fortranobject.c'

# ---------------------------------------------------------------------
# Fortran sources, one program unit per file.
# ---------------------------------------------------------------------

fortran_sources = files(
  # convex quadratic programming, Goldfarb-Idnani
  'ql.f',
  'qldual.f',
  'qlchol.f',
  'qladdc.f',
  'qldrop.f',
  'qlnorm.f',
  'qlslk.f',
  # sequential quadratic programming
  'nlpqlp.f',
  'nlpqlw.f',
  'nlpqlpq.f',
  'nlpql.f',
  'nlpqly.f',
  'nlpqlyw.f',
  'nlpqlb.f',
  'nlpqlbw.f',
  'nlpqlbs.f',
  'nlpqlbq.f',
  'nlpqlg.f',
  'nlpqlgw.f',
  'nlpqlgq.f',
  'nlpqlf.f',
  'nlpqlfw.f',
  'nlpqlfq.f',
  # data fitting
  'nldftw.f',
  'nldftf.f',
  'nldftg.f',
  'nlplsq.f',
  'nlplsqq.f',
  'nlplsx.f',
  'nlplsxq.f',
  'nlpl1.f',
  'nlpl1q.f',
  'nlpinf.f',
  'nlpinfq.f',
  'nlpmmx.f',
  'nlpmmxq.f',
  # multicriteria optimization
  'nlpjob.f',
  'nlpjbw.f',
  'nlpjbc.f',
  'nlpjbf.f',
  'nlpjbg.f',
  # auxiliary routines of the SQP algorithm
  'nlmert.f',
  'nldphi.f',
  'nlpen.f',
  'nlrest.f',
  'nlbfgs.f',
  'nlldlf.f',
  'nlldlc.f',
  'nlprnt.f',
)

python_sources = files(
  '__init__.py',
  '_drivers.py',
  '_optimize.py',
  '_problem.py',
  '_qp.py',
  '_status.py',
  '_workspace.py',
)

py_mod_name = '_nlpql'

f2py_target = custom_target(
  py_mod_name + '_f2py',
  input: files('nlpql.pyf'),
  output: [py_mod_name + 'module.c', py_mod_name + '-f2pywrappers.f'],
  command: [f2py, '@INPUT@', '--lower', '--build-dir', meson.current_build_dir()]
)

# The extension module mixes C and Fortran, so meson drives the link
# step with the C compiler.  Not every runtime flag of gfortran is
# understood by that driver, therefore each of them is probed before it
# is used.  Linking the GCC runtimes statically keeps the wheel free of
# any dependency on the compiler installation.
cc = meson.get_compiler('c')

# Meson appends '-lgfortran' after the arguments given here, so relying on
# '-static-libgfortran' alone is not enough: the C driver silently ignores
# that flag on some toolchains and the trailing '-lgfortran' then resolves
# against the import library, leaving the extension with a dependency on
# libgfortran-*.dll that is not on the DLL search path at import time.
# Pulling the runtimes in explicitly inside a '-Bstatic' block resolves
# every Fortran symbol from the static archives first, so the later
# '-lgfortran' has nothing left to add.
if host_machine.system() == 'windows'
  fortran_link_args = cc.get_supported_link_arguments([
    '-static-libgfortran',
    '-static-libgcc',
    '-static-libquadmath',
    '-Wl,-Bstatic',
    '-lgfortran',
    '-lquadmath',
    '-lwinpthread',
    '-Wl,-Bdynamic',
  ])
else
  fortran_link_args = cc.get_supported_link_arguments([
    '-static-libgfortran',
    '-static-libgcc',
  ])
endif

py.extension_module(
  py_mod_name,
  fortran_sources + [f2py_target, fortranobject_c],
  c_args: ['-I' + incdir_numpy, '-I' + incdir_f2py],
  link_args: fortran_link_args,
  dependencies: py_dep,
  subdir: 'nlpql',
  install: true
)

py.install_sources(
  python_sources,
  subdir: 'nlpql'
)
