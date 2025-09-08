## ----------------------
## Trajectory generator - a script to automatically create hundreds of trajectory calculations
## ----------------------
##
## Gen-FSSH.py 2019-2022 Jingbai Li
## New version Oct 13 2022 Jingbai Li
## Upgrade to multiprocessing Nov 5 2022 Jingbai Li
## Support user-defined shell commands Nov 28 Jingbai Li
## Support openqp and lammps Sep 8 2025 Jingbai Li

import sys
import os
import shutil
import json
import copy
import multiprocessing
import numpy as np
## import initial condition sampling module
try:
    from PyRAI2MD.Utils.sampling import sampling
except ModuleNotFoundError:
    exit('PyRAI2MD is not installed, stop sampling')


def main(argv):
    ##  This is the main function
    ##  It read all options from a sampling file and
    ##  perform initial condition sampling

    usage = """

    Trajectory Generator

    Usage:
      python3 traj_generator.py sampling or
      python3 traj_generator for help

    a sampling file contains the following parameters

      cpus          number of cpu for parallel generation
      notraj        0
      input         file_name.freq.molden
      seed          -1
      temp          298.15
      method        wigner
      partition     normal
      time          4-23:59:59
      memory        3000
      nodes         1
      cores         28
      jobs          28
      index         1
      prog          molcas
      molcas        /path/to/Molcas
      newton        /path/to/NX-2.2-B08
      bagel         /path/to/Bagel
      openqp        /path/to/openqp
      lammps        /path/to/lammps
      lib_blas      /path/to/blas
      lib_lapack    /path/to/lapack
      lib_scalapack /path/to/scalapack
      lib_boost     /path/to/Boost
      mkl           /path/to/mkl
      mpi           /path/to/mpi
      >shell
      additional shell commands ...
      >shell

    For more information, please see traj-generator-readme.txt

    """

    ## defaults parameters
    inputs = ''
    cpus = 1
    notraj = 0
    iseed = 1
    temp = 298.15
    dist = 'wigner'
    prog = 'molcas'
    slpt = 'normal'
    sltm = '1-00:00:00'
    slmm = 100
    slnd = 1
    slcr = 1
    sljb = 1
    slin = 1
    repeat = 0
    tomlcs = '/public/software/app/molcas'
    tontx = '/share/apps/NX-2.4-B06 '
    tobgl = '/public/software/app/bagel'
    lbbls = ''
    lblpk = ''
    lbslp = ''
    lbbst = ''
    tomkl = ''
    tompi = ''
    toxtb = '/public/software/app/xtb-6.7.1'
    toorca = '/public/software/app/orca_6_0_1_linux_x86-64_shared_openmpi416'
    tooqp = '/public/software/app/openqp'
    tolmp = '/public/software/app/lammps'

    if len(argv) <= 1:
        exit(usage)

    with open(argv[1]) as inp:
        infile = inp.read().split('>shell')
    if len(infile) == 1:
        shell = ''
        inputfile = infile[0].splitlines()
    elif len(infile) == 3:
        shell = infile[1]
        inputfile = infile[0].splitlines() + infile[2].splitlines()
    else:
        shell = None
        inputfile = None
        exit('\n SyntaxError: require 2 >shell delimiters, but found %s\n' % (len(infile) - 1))

    for line in inputfile:
        if len(line.split()) < 2:
            continue
        key = line.split()[0].lower()
        if 'input' == key:
            inputs = line.split()[1]
        elif 'cpus' == key:
            cpus = int(line.split()[1])
        elif 'notraj' == key:
            notraj = int(line.split()[1])
        elif 'seed' == key:
            iseed = int(line.split()[1])
        elif 'temp' == key:
            temp = float(line.split()[1])
        elif 'method' == key:
            dist = line.split()[1].lower()
        elif 'prog' == key:
            prog = line.split()[1].lower()
        elif 'partition' == key:
            slpt = line.split()[1]
        elif 'time' == key:
            sltm = line.split()[1]
        elif 'memory' == key:
            slmm = int(line.split()[1])
        elif 'nodes' == key:
            slnd = int(line.split()[1])
        elif 'cores' == key:
            slcr = int(line.split()[1])
        elif 'jobs' == key:
            sljb = int(line.split()[1])
        elif 'index' == key:
            slin = int(line.split()[1])
        elif 'repeat' == key:
            repeat = int(line.split()[1])
        elif 'molcas' == key:
            tomlcs = line.split()[1]
        elif 'newton' == key:
            tontx = line.split()[1]
        elif 'bagel' == key:
            tobgl = line.split()[1]
        elif 'lib_blas' == key:
            lbbls = line.split()[1]
        elif 'lib_lapack' == key:
            lblpk = line.split()[1]
        elif 'lib_scalapack' == key:
            lbslp = line.split()[1]
        elif 'lib_boost' == key:
            lbbst = line.split()[1]
        elif 'mkl' == key:
            tomkl = line.split()[1]
        elif 'mpi' == key:
            tompi = line.split()[1]
        elif 'xtb' == key:
            toxtb = line.split()[1]
        elif 'orca' == key:
            toorca = line.split()[1]
        elif 'openqp' == key:
            tooqp = line.split()[1]
        elif 'lammps' == key:
            tolmp = line.split()[1]

    if inputs is not None and os.path.exists(inputs):
        print('\n>>> %s' % inputs)
    else:
        print('\n!!! File %s not found !!!' % inputs)
        print(usage)
        print('!!! File %s not found !!!' % inputs)
        exit()

    iformat = inputs.split('.')[-1]
    init_file = inputs.split('.')[-2]
    inputs = inputs.split('.')[0]

    if notraj == 0:
        if prog == 'molcas':
            if not os.path.exists('%s.inp' % inputs):
                print('\n!!! Molcas template input %s.inp not found !!!' % inputs)
                print(usage)
                print('!!! Molcas template input %s.inp not found !!!\n' % inputs)
                exit()
            if not os.path.exists('%s.StrOrb' % inputs) and not os.path.exists('%s.JobIph' % inputs):
                print('\n!!! Molcas orbital file %s.StrOrb or JobIph not found !!!' % inputs)
                print(usage)
                print('!!! Molcas orbital file %s.StrOrb or JobIph not found !!!\n' % inputs)
                exit()
        elif prog == 'bagel':
            if not os.path.exists('%s.bagel' % inputs):
                print('\n!!! Bagel template input %s.bagel not found !!!' % inputs)
                print(usage)
                print('\n!!! Bagel template input %s.bagel not found !!!' % inputs)
                exit()
            if not os.path.exists('%s.archive' % inputs):
                print('\n!!! Bagel orbital file %s.archive not found !!!' % inputs)
                print(usage)
                print('\n!!! Bagel orbital file %s.archive not found !!!' % inputs)
                exit()
        elif prog == 'nxbagel':
            if not os.path.exists('control.dyn'):
                print('\n!!! NewtonX: control.dyn not found !!!')
                print(usage)
                print('!!! NewtonX: control.dyn not found !!!')
                exit()
            if not os.path.exists('bagelinput.basis.inp'):
                print('\n!!! Bagel: bagelinput.basis.inp not found !!!')
                print(usage)
                print('!!! Bagel: bagelinput.basis.inp not found !!!')
                exit()
            if not os.path.exists('bagelinput.part1.inp') or not os.path.exists(
                    'bagelinput.part2.inp') or not os.path.exists('bagelinput.part3.inp'):
                print('\n!!! Bagel: bagelinput.part1-3.inp not found !!!')
                print(usage)
                print('!!! Bagel: bagelinput.part1-3.inp not found !!!')
                exit()
        elif prog == 'pyrai2md':
            if not os.path.exists('input'):
                print('\n!!! PyRAI2MD input not found !!!')
                print(usage)
                print('!!! PyRAI2MD input not found !!!')
                exit()
        elif prog == 'pyrai2mdnn':
            if not os.path.exists('input'):
                print('\n!!! PyRAI2MD input not found !!!')
                print(usage)
                print('!!! PyRAI2MD input not found !!!')
                exit()
        elif prog == 'pyrai2mdmolcas' or prog == 'pyrai2mdhybrid':
            if not os.path.exists('input'):
                print('\n!!! PyRAI2MD input not found !!!')
                print(usage)
                print('!!! PyRAI2MD input not found !!!')
                exit()
            if not os.path.exists('%s.molcas' % inputs):
                print('\n!!! PyRAI2MD molcas template not found !!!')
                print(usage)
                print('!!! PyRAI2MD molcas template not found !!!')
                exit()
            if not os.path.exists('%s.StrOrb' % inputs) and not os.path.exists('%s.JobIph' % inputs):
                print('\n!!! PyRAI2MD molcas orbital file StrOrb or JobIph not found !!!')
                print(usage)
                print('!!! PyRAI2MD molcas orbital file StrOrb or JobIph not found !!!')
                exit()
        elif prog == 'fromage':
            if not os.path.exists('fromage.in'):
                print('\n!!! fromage input not found !!!')
                print(usage)
                print('!!! fromage input not found !!!')
                exit()
            if not os.path.exists('mh.temp'):
                print('\n!!! fromage mh.temp not found !!!')
                print(usage)
                print('!!! fromage mh.temp not found !!!')
                exit()
            if not os.path.exists('%s.StrOrb' % input):
                print('\n!!! fromage orbital file %s.StrOrb not found !!!' % inputs)
                print(usage)
                print('!!! fromage orbital file %s.StrOrb found !!!' % inputs)
                exit()
            if not os.path.exists('rl.temp'):
                print('\n!!! fromage rl.temp not found !!!')
                print(usage)
                print('!!! fromage rl.temp not found !!!')
                exit()
            if not os.path.exists('ml.temp'):
                print('\n!!! fromage ml.temp not found !!!')
                print(usage)
                print('!!! fromage ml.temp not found !!!')
                exit()
            if not os.path.exists('xtb.input'):
                print('\n!!! fromage xtb.input not found !!!')
                print(usage)
                print('!!! fromage xtb.input not found !!!')
                exit()
            if not os.path.exists('xtb_charge.pc'):
                print('\n!!! fromage xtb_charge.pc  not found !!!')
                print(usage)
                print('!!! fromage xtb_charge.pc not found !!!')
                exit()
        elif prog == 'orca':
            if not os.path.exists('%s.orca' % inputs):
                print('\n!!! orca input not found !!!')
                print(usage)
                print('!!! orca input not found !!!')
                exit()
        elif prog == 'openqp':
            if not os.path.exists('%s.openqp' % inputs):
                print('\n!!! OpenQP input not found !!!')
                print(usage)
                print('!!! OpenQP input not found !!!')
                exit()
        elif prog == 'lammps':
            if not os.path.exists('%s.in' % inputs):
                print('\n!!! LAMMPS input not found !!!')
                print(usage)
                print('!!! LAMMPS input not found !!!')
            if not os.path.exists('%s.data' % inputs):
                print('\n!!! LAMMPS data not found !!!')
                print(usage)
                print('!!! LAMMPS data not found !!!')
                exit()
            if not os.path.exists('%s.in.init' % inputs):
                print('\n!!! LAMMPS in.init file not found !!!')
                print(usage)
                print('!!! LAMMPS in.init file not found !!!')
            if not os.path.exists('%s.in.settings' % inputs):
                print('\n!!! LAMMPS in.settings file not found !!!')
                print(usage)
                print('!!! LAMMPS in.settings file not found !!!')
        else:
            print('\n!!! Program %s not found !!!' % prog)
            print(usage)
            print('!!! Program %s not found !!!' % prog)
            exit()

    else:
        print(' Only perform initial condition sampling')

    callsample = ['molden', 'bagel', 'g16', 'orca']
    skipsample = ['newtonx', 'xyz', 'xz2']
    nesmb = int(sljb * slnd)

    if iformat in callsample:
        print("""
        Start Sampling
  ------------------------------------------------
    Seed         %10d
    Method       %10s
    Trajectories %10d = %3d Nodes X %3d Jobs
    Temperature  %10.2f
    """ % (iseed, dist, nesmb, slnd, sljb, temp))

    elif iformat in skipsample:
        print("""
        Read Sampled Initial Conditions

    """)

    if repeat == 0:
        if iformat == 'xyz' and init_file != 'init':
            ensemble = read_xyzlist(inputs, nesmb)
        else:
            ensemble = sampling(inputs, nesmb, iseed, temp, dist, iformat)  # generate initial conditions
    else:
        print("Reuse the first sampled condition %s times" % nesmb)
        ensemble = sampling(inputs, 1, iseed, temp, dist, iformat)  # generate initial conditions
        ensemble = [ensemble[0] for _ in range(nesmb)]

    if notraj != 0:
        print('\n\n Done\n')
        exit()

    if prog == 'molcas':
        gen_molcas(cpus, ensemble, inputs, slpt, sltm, slmm, slnd, slcr, sljb, slin, tomlcs, iformat, shell)
    elif prog == 'bagel':
        gen_bagel(cpus, ensemble, inputs, slpt, sltm, slmm, slnd, slcr, sljb, slin, tobgl,
                  lbbls, lblpk, lbslp, lbbst, tomkl, tompi, shell)
    elif prog == 'nxbagel':
        gen_nxbagel(cpus, ensemble, inputs, slpt, sltm, slmm, slnd, slcr, sljb, slin, tontx, tobgl,
                    lbbls, lblpk, lbslp, lbbst, tomkl, tompi, shell)
    elif prog == 'pyrai2md':
        gen_pyrai2md(cpus, ensemble, inputs, slpt, sltm, slmm, slnd, slcr, sljb, slin, 'no', iformat, shell)
    elif prog == 'pyrai2mdnn':
        gen_pyrai2md(cpus, ensemble, inputs, slpt, sltm, slmm, slnd, slcr, sljb, slin, 'nn', iformat, shell)
    elif prog == 'pyrai2mdmolcas':
        gen_pyrai2md(cpus, ensemble, inputs, slpt, sltm, slmm, slnd, slcr, sljb, slin, 'molcas', iformat, shell)
    elif prog == 'pyrai2mdbagel':
        gen_pyrai2md(cpus, ensemble, inputs, slpt, sltm, slmm, slnd, slcr, sljb, slin, 'bagel', iformat, shell)
    elif prog == 'pyrai2mdorca':
        gen_pyrai2md(cpus, ensemble, inputs, slpt, sltm, slmm, slnd, slcr, sljb, slin, 'orca', iformat, shell)
    elif prog == 'pyrai2mdhybrid':
        gen_pyrai2md(cpus, ensemble, inputs, slpt, sltm, slmm, slnd, slcr, sljb, slin, 'hybrid', iformat, shell)
    elif prog == 'pyrai2mdoqp':
        gen_pyrai2md(cpus, ensemble, inputs, slpt, sltm, slmm, slnd, slcr, sljb, slin, 'oqp', iformat, shell)
    elif prog == 'fromage':
        gen_fromage(cpus, ensemble, inputs, slpt, sltm, slmm, slnd, slcr, sljb, slin, tomlcs, toxtb, iformat, shell)
    elif prog == 'orca':
        gen_orca(cpus, ensemble, inputs, slpt, sltm, slmm, slnd, slcr, sljb, slin, toorca, iformat, shell)
    elif prog == 'openqp':
        gen_oqp(cpus, ensemble, inputs, slpt, sltm, slmm, slnd, slcr, sljb, slin, tooqp, iformat, shell)
    elif prog == 'lammps':
        gen_lmp(cpus, ensemble, inputs, slpt, sltm, slmm, slnd, slcr, sljb, slin, tolmp, iformat, shell)

def gen_molcas(cpus, ensemble, inputs, slpt, sltm, slmm, slnd, slcr, sljb, slin, tomlcs, iformat, shell):
    ## This function will group Molcas calculations to individual runset
    ## this function will call molcas_batch and molcas to prepare files

    marks = []
    if os.path.exists('%s.basis' % inputs):
        with open('%s.basis' % inputs) as atommarks:
            marks = atommarks.read().splitlines()
            natom = int(marks[0])
            marks = marks[2:2 + natom]

    in_temp = open('%s.inp' % inputs, 'r').read()
    if os.path.exists('%s.StrOrb' % inputs):
        in_orb = open('%s.StrOrb' % inputs, 'r').read()
    else:
        in_orb = None
    if os.path.exists('%s.JobIph' % inputs):
        jobiph = '%s.JobIph' % inputs
    else:
        jobiph = None
    if os.path.exists('%s.key' % inputs):
        qmmmkey = '%s.key' % inputs
    else:
        qmmmkey = None

    in_path = os.getcwd()

    runall = ''
    variables_wrapper = []
    for j in range(slnd):
        runall += 'sbatch runset-%d.sh\n' % (j + 1)
        start = slin + j * sljb
        end = start + sljb - 1
        for i in range(sljb):
            if iformat != 'xz':
                # unpack initial condition to xyz and velocity
                in_xyz, in_velo = Unpack(ensemble[i + j * sljb], 'molcas')
                if len(marks) > 0:
                    in_xyz = Markatom(in_xyz, marks)
            else:
                in_xyz, in_velo = UnpackXZ(ensemble[i + j * sljb])
            inputname = '%s-%s' % (inputs, i + start)
            inputpath = '%s/%s' % (in_path, inputname)
            # prepare calculations
            variables_wrapper.append([
                inputname, inputpath, slpt, slmm, in_temp, in_orb, jobiph, qmmmkey, in_xyz, in_velo, tomlcs, shell
            ])

        batch = molcas_batch(inputs, j, start, end, in_path, slcr, sltm, slpt, slmm, tomlcs, shell)

        with open('./runset-%d.sh' % (j + 1), 'w') as run:
            run.write(batch)

        os.system("chmod 777 runset-%d.sh" % (j + 1))

    task = len(variables_wrapper)
    cpus = min([task, cpus])
    pool = multiprocessing.Pool(processes=cpus)
    n = 0
    for _ in pool.imap_unordered(molcas, variables_wrapper):
        n += 1
        sys.stdout.write('CPU: %3d generating trajectory: %d/%d\r' % (cpus, n, task))
    pool.close()

    with open('./runall.sh', 'w') as out:
        out.write(runall)

    os.system("chmod 777 runall.sh")
    print('\n\n Done\n')


def molcas_batch(inputs, j, start, end, in_path, slcr, sltm, slpt, slmm, tomlcs, shell):
    ## This function will be called by gen_molcas function
    ## This function generates runset for MolCas calculation

    ## copy necessary module for to Molcas Tools, not useful
    # if restart != 0 and os.path.exists('%s/Tools/TSHrestart.py' % tomlcs) == False:
    #     shutil.copy2('%s/bin/TSHrestart.py' % pyqd, '%s/Tools/TSHrestart.py' % tomlcs)
    # if initex != 0 and os.path.exists('%s/Tools/InitEx.py' % tomlcs) == False:
    #     shutil.copy2('%s/bin/InitEx.py' % pyqd, '%s/Tools/InitEx.py' % tomlcs)
    #     if initex == 0:
    #         pri = ''
    #     else:
    #         pri = """
    #   if [ "$STEP" == "0" ]
    #   then
    #     python3 $MOLCAS/Tools/InitEx.py prep $INPUT.inp
    #     cd INITEX
    #     $MOLCAS/bin/pymolcas -f $INPUT.inp -b 1
    #     cd ../
    #     python3 $MOLCAS/Tools/InitEx.py read $INPUT.inp
    #   fi
    # """
    #     if restart == 0:
    #         add = ' '
    #         addend = ' '
    #     else:
    #         add = """
    #   python3 $MOLCAS/Tools/TSHrestart.py PROG
    # """
    #         addend = """
    # STEP=`tail -n1 PROG|awk '{print $1}'`
    # if [ "$STEP" -lt "$MAX" ]
    # then
    # cd ../
    # sbatch runset-%d.sh
    # fi
    # """ % (j + 1)

    pri = ''
    add = ''
    addend = ''
    restart = 0
    batch = """#!/bin/sh
## script for OpenMalCas
#SBATCH --nodes=1
#SBATCH --ntasks=%d
#SBATCH --time=%s
#SBATCH --job-name=%s-%d
#SBATCH --partition=%s
#SBATCH --mem=%dmb
#SBATCH --output=%%j.o.slurm
#SBATCH --error=%%j.e.slurm
%s
export MOLCAS_NPROCS=1
export MOLCAS_MEM=%d
export MOLCAS_PRINT=2
export OMP_NUM_THREADS=1
export MOLCAS=%s
export TINKER=$MOLCAS/tinker-6.3.3/bin
export PATH=$MOLCAS/bin:$PATH

echo $SLURM_JOB_NAME

if [ -d "/srv/tmp" ]
then
 export LOCAL_TMP=/srv/tmp
else
 export LOCAL_TMP=/tmp
fi

RunMolcas(){
  if [ -a "PROG" ]
  then
    STEP=`tail -n1 PROG|awk '{print $1}'`
  else
    STEP=0
    echo "$INPUT $MAX" > PROG
    echo "$STEP" >> PROG
  fi
%s
  $MOLCAS/bin/pymolcas -f $INPUT.inp -b 1
  STEP=`expr $STEP + 1`
  echo "$STEP" >> PROG
%s
  rm -r $MOLCAS_WORKDIR/$MOLCAS_PROJECT
}

MAX=%s

for ((i=%d;i<=%d;i++))
do
  export INPUT="%s-$i"
  export WORKDIR="%s/%s-$i"
  export MOLCAS_PROJECT=$INPUT
  export MOLCAS_WORKDIR=$LOCAL_TMP/$USER/$SLURM_JOB_ID
  mkdir -p $MOLCAS_WORKDIR/$MOLCAS_PROJECT
  cd $WORKDIR
  RunMolcas &
  sleep 5
done
wait
rm -r $MOLCAS_WORKDIR

%s
""" % (
        slcr, sltm, inputs, j + 1, slpt, int(slmm * slcr * 1.333), shell, slmm, tomlcs,
        pri, add, restart + 1, start, end, inputs, in_path, inputs, addend
    )

    return batch


def molcas(var):
    ## This function prepares MolCas calculation
    ## It generates .inp .StrOrb .xyz .velocity.xyz
    ## This function generates a backup slurm batch file for each calculation
    inputname, inputpath, slpt, slmm, in_temp, in_orb, jobiph, qmmmkey, in_xyz, in_velo, tomlcs, shell = var
    if not os.path.exists('%s' % inputpath):
        os.makedirs('%s' % inputpath)

    runscript = """#!/bin/sh
## backup script for OpenMalCas
## $INPUT and $WORKDIR do not belong to OpenMolCas
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=23:50:00
#SBATCH --job-name=%s
#SBATCH --partition=%s
#SBATCH --mem=%dmb
#SBATCH --output=%%j.o.slurm
#SBATCH --error=%%j.e.slurm
%s
if [ -d "/srv/tmp" ]
then
 export LOCAL_TMP=/srv/tmp
else
 export LOCAL_TMP=/tmp
fi

export INPUT=%s
export WORKDIR=%s
export MOLCAS_NPROCS=1
export MOLCAS_MEM=%d
export MOLCAS_PRINT=2
export MOLCAS_PROJECT=$INPUT
export OMP_NUM_THREADS=1
export MOLCAS=%s
export TINKER=$MOLCAS/tinker-6.3.3/bin
export MOLCAS_WORKDIR=$LOCAL_TMP/$USER/$SLURM_JOB_ID
export PATH=$MOLCAS/bin:$PATH

mkdir -p $MOLCAS_WORKDIR/$MOLCAS_PROJECT
cd $WORKDIR
$MOLCAS/bin/pymolcas -f $INPUT.inp -b 1
rm -r $MOLCAS_WORKDIR
""" % (inputname, slpt, int(slmm * 1.333), shell, inputname, inputpath, slmm, tomlcs)

    with open('%s/%s.inp' % (inputpath, inputname), 'w') as out:
        out.write(in_temp)

    with open('%s/%s.sh' % (inputpath, inputname), 'w') as out:
        out.write(runscript)

    if in_orb is not None:
        with open('%s/%s.StrOrb' % (inputpath, inputname), 'w') as out:
            out.write(in_orb)

    if jobiph is not None:
        shutil.copy2(jobiph, '%s/%s.JobIph' % (inputpath, inputname))

    if qmmmkey is not None:
        shutil.copy(qmmmkey, '%s/%s.key' % (inputpath, inputname))

    with open('%s/%s.xyz' % (inputpath, inputname), 'w') as out:
        out.write(in_xyz)

    with open('%s/%s.velocity.xyz' % (inputpath, inputname), 'w') as out:
        out.write(in_velo)

    os.system("chmod 777 %s/%s.sh" % (inputpath, inputname))


def Markatom(xyz, marks):
    ## This function marks atoms for different basis set specification of Molcas
    ## here xyz, e, x, y, z are all strings

    xyz = xyz.splitlines()
    new_xyz = '%s\n%s\n' % (xyz[0], xyz[1])
    for n, line in enumerate(xyz[2:]):
        e, x, y, z = line.split()[0:4]
        e = marks[n].split()[0]
        new_xyz += '%-5s%30s%30s%30s\n' % (e, x, y, z)

    return new_xyz


def gen_bagel(cpus, ensemble, inputs, slpt, sltm, slmm, slnd, slcr, sljb, slin, tobgl, lbbls, lblpk, lbslp,
              lbbst, tomkl, tompi, shell):
    ## This function generate bagel calculations in individual runset
    ## This function will call bagel_batch and bagel to prepare files

    if os.path.exists('%s.charge' % inputs):
        with open('%s.charge' % inputs, 'r') as infile:
            charge = infile.read().splitlines()
            nq = int(charge[0])
            charge = [line.split()[0: 4] for line in charge[2: 2 + nq]]
    else:
        charge = None

    with open('%s.bagel' % inputs, 'r') as infile:
        in_temp = json.load(infile)

    in_path = os.getcwd()
    variables_wrapper = []
    runall = ''
    for j in range(slnd):
        runall += 'sbatch runset-%d.sh\n' % (j + 1)
        start = slin + j * sljb
        end = start + sljb - 1
        for i in range(sljb):
            in_xyz, in_velo = Unpack(ensemble[i + j * sljb], 'bagel')  # unpack initial condition to xyz and velocity
            inputname = '%s-%s' % (inputs, i + start)
            inputpath = '%s/%s' % (in_path, inputname)
            variables_wrapper.append([
                inputs, inputname, inputpath, slcr, sljb, sltm, slpt, slmm, in_temp, in_xyz, in_velo, charge, tobgl,
                lbbls, lblpk, lbslp, lbbst, tomkl, tompi, shell
            ])  # prepare calculations

        batch = bagel_batch(inputs, j, start, end, in_path, slcr, sljb, sltm, slpt, slmm, tobgl, lbbls, lblpk,
                            lbslp, lbbst, tomkl, tompi, shell)

        with open('./runset-%d.sh' % (j + 1), 'w') as out:
            out.write(batch)

        os.system("chmod 777 runset-%d.sh" % (j + 1))

    task = len(variables_wrapper)
    cpus = min([task, cpus])
    pool = multiprocessing.Pool(processes=cpus)
    n = 0
    for _ in pool.imap_unordered(bagel, variables_wrapper):
        n += 1
        sys.stdout.write('CPU: %3d generating trajectory: %d/%d\r' % (cpus, n, task))
    pool.close()

    with open('./runall.sh', 'w') as out:
        out.write(runall)

    os.system("chmod 777 runall.sh")
    print('\n\n Done\n')


def bagel_batch(inputs, j, start, end, in_path, slcr, sljb, sltm, slpt, slmm, tobgl, lbbls, lblpk, lbslp,
                lbbst, tomkl, tompi, shell):
    ## This function will be called by gen_bagel
    ## This function generates runset for Bagel calculation

    ### Note, this line doesn't check if slcr and sljb are appropriate for parallelization, be careful!!!
    bagelpal = int(slcr / sljb)

    batch = """#!/bin/sh
## script for BAGEL
#SBATCH --nodes=1
#SBATCH --ntasks=%d
#SBATCH --time=%s
#SBATCH --job-name=%s-%d
#SBATCH --partition=%s
#SBATCH --mem=%dmb
#SBATCH --output=%%j.o.slurm
#SBATCH --error=%%j.e.slurm
%s

export MKL_DEBUG_CPU_TYPE=5

# run BAGEL in multi thread mode
export BAGELPAL=%s
export BAGEL_NUM_THREADS=$BAGELPAL
export OMP_NUM_THREADS=$BAGELPAL
export MKL_NUM_THREADS=$BAGELPAL

export BAGEL=%s/bin/BAGEL
export BLAS_LIB=%s
export LAPACK_LIB=%s
export SCALAPACK_LIB=%s
export BOOST_LIB=%s/lib

#source %s/setvars.sh ## This is not used if you don't use intel/oneAPI for mkl and mpi
#export MPI=%s ## mpi is not used here
#export LD_LIBRARY_PATH$MPI/lib:$LD_LIBRARY_PATH
#export PATH=$MPI/bin:$PATH

export LD_LIBRARY_PATH=$BAGEL/lib:$BLAS_LIB:$LAPACK_LIB:$SCALAPACK_LIB:$BOOST_LIB:$LD_LIBRARY_PATH

echo $SLURM_JOB_NAME

for ((i=%d;i<=%d;i++))
do
  export INPUT=%s-$i
  export WORKDIR=%s/%s-$i
  cd $WORKDIR
  $BAGEL $INPUT.json > $INPUT.log &
  sleep 5
done
wait
""" % (
        slcr, sltm, inputs, j + 1, slpt, int(slmm * slcr * 1.1), shell, bagelpal, tobgl, lbbls, lblpk,
        lbslp, lbbst, tomkl, tompi, start, end, inputs, in_path, inputs)

    return batch


def bagel(var):
    ## This function prepares NxBagel calculation
    ## It generates TRAJECTORIES/TRAJ#/JOB_NAD, geom, veloc, controd.dyn for NxBagel calculations
    ## This function generates a backup slurm batch file for each calculation

    inputs, inputname, inputpath, slcr, sljb, sltm, slpt, slmm, in_temp, in_xyz, in_velo, charge, tobgl, lbbls, lblpk, \
        lbslp, lbbst, tomkl, tompi, shell = var

    if not os.path.exists('%s' % inputpath):
        os.makedirs('%s' % inputpath)

    ### Note, this line doesn't check if slcr and sljb are appropriate for parallelization, be careful!!!
    bagelpal = int(slcr / sljb)

    runscript = """#!/bin/sh
## script for BAGEL
#SBATCH --nodes=1
#SBATCH --ntasks=%s
#SBATCH --time=%s
#SBATCH --job-name=%s
#SBATCH --partition=%s
#SBATCH --mem=%dmb
#SBATCH --output=%%j.o.slurm
#SBATCH --error=%%j.e.slurm
%s

# run BAGEL in multi thread mode
export BAGELPAL=%s
export BAGEL_NUM_THREADS=$BAGELPAL
export OMP_NUM_THREADS=$BAGELPAL
export MKL_NUM_THREADS=$BAGELPAL

export BAGEL=%s/bin/BAGEL
export BLAS_LIB=%s
export LAPACK_LIB=%s
export SCALAPACK_LIB=%s
export BOOST_LIB=%s/lib

#source %s/setvars.sh ## This is not used if you don't use intel/oneAPI for mkl and mpi
#export MPI=%s ## mpi is not used here
#export LD_LIBRARY_PATH$MPI/lib:$LD_LIBRARY_PATH
#export PATH=$MPI/bin:$PATH

export LD_LIBRARY_PATH=$BAGEL/lib:$BLAS_LIB:$LAPACK_LIB:$SCALAPACK_LIB:$BOOST_LIB:$LD_LIBRARY_PATH

export INPUT=%s
export WORKDIR=%s
cd $WORKDIR
$BAGEL $INPUT.json > $INPUT.log
""" % (bagelpal, sltm, inputname, slpt, int(slmm * 1.1), shell, bagelpal, tobgl, lbbls, lblpk, lbslp, lbbst,
       tomkl, tompi, inputname, inputpath)

    with open('%s/%s.sh' % (inputpath, inputname), 'w') as out:
        out.write(runscript)

    jxyz = write_json_xyz(in_xyz, charge)
    si_input = in_temp.copy()
    si_input['bagel'][0]['geometry'] = jxyz
    si_input['bagel'][1]['file'] = inputname

    with open('%s/%s.json' % (inputpath, inputname), 'w') as out:
        json.dump(si_input, out, indent=2)

    if os.path.exists('%s.archive' % inputs) and not os.path.exists('%s/%s.archive' % (inputpath, inputname)):
        shutil.copy2('%s.archive' % inputs, '%s/%s.archive' % (inputpath, inputname))

    os.system("chmod 777 %s/%s.sh" % (inputpath, inputname))


def write_json_xyz(x, q=None):
    ## write coordinate file
    ## convert xyz from array to bagel format (Bohr)

    a2b = 1 / 0.529177249  # angstrom to bohr
    jxyz = []
    for line in x:
        e, x, y, z = line
        jxyz.append({"atom": e, "xyz": [float(x) * a2b, float(y) * a2b, float(z) * a2b]})

    ## convert point charge from array to bagel format
    if q:
        for charge in q:
            jxyz.append(
                {"atom": "Q",
                 "xyz": [float(charge[1]) * a2b, float(charge[2]) * a2b, float(charge[3]) * a2b],
                 "charge": charge[0]}
            )

    return jxyz


def gen_nxbagel(cpus, ensemble, inputs, slpt, sltm, slmm, slnd, slcr, sljb, slin, tontx, tobgl, lbbls, lblpk, lbslp,
                lbbst, tomkl, tompi, shell):
    ## This function groups nxbagel calculations in individual runset
    ## This function will call nxbagel_batch and nxbagel to prepare files
    ## This function need control.dyn for NxBagel calculation, therm.inp, sh.inp and jiri.inp are optional

    in_orb = ''
    in_jiri = ''
    in_sh = ''
    in_therm = ''
    in_bagel0 = open('bagelinput.basis.inp', 'r').read()
    in_bagel1 = open('bagelinput.part1.inp', 'r').read()
    in_bagel2 = open('bagelinput.part2.inp', 'r').read()
    in_bagel3 = open('bagelinput.part3.inp', 'r').read()
    in_control = open('control.dyn', 'r').read()

    if os.path.exists('%s.archive' % inputs):
        in_orb = '%s.archive' % inputs
    if os.path.exists('jiri.inp'):
        in_jiri = open('jiri.inp', 'r').read()
    if os.path.exists('sh.inp'):
        in_sh = open('sh.inp', 'r').read()
    if os.path.exists('therm.inp'):
        in_therm = open('therm.inp', 'r').read()

    in_temp = {
        'jiri': in_jiri,
        'sh': in_sh,
        'therm': in_therm,
        'orb': in_orb,
        'control': in_control,
        'basis': in_bagel0,
        'part1': in_bagel1,
        'part2': in_bagel2,
        'part3': in_bagel3
    }

    in_path = os.getcwd()
    variables_wrapper = []
    runall = ''
    for j in range(slnd):
        runall += 'sbatch runset-%d.sh\n' % (j + 1)
        start = slin + j * sljb
        end = start + sljb - 1
        for i in range(sljb):
            in_xyz, in_velo = Unpack(ensemble[i + j * sljb], 'newton')  # unpack initial condition to xyz and velocity
            inputname = '%s-%s' % (inputs, i + start)
            inputpath = '%s/%s' % (in_path, inputname)
            variables_wrapper.append([
                inputname, inputpath, slcr, sljb, sltm, slpt, slmm, in_temp, in_xyz, in_velo, tontx, tobgl,
                lbbls, lblpk, lbslp, lbbst, tomkl, tompi, shell
            ])  # prepare calculations

        batch = nxbagel_batch(inputs, j, start, end, in_path, slcr, sljb, sltm, slpt, slmm, tontx, tobgl, lbbls, lblpk,
                              lbslp, lbbst, tomkl, tompi, shell)

        with open('./runset-%d.sh' % (j + 1), 'w') as out:
            out.write(batch)

        os.system("chmod 777 runset-%d.sh" % (j + 1))

    task = len(variables_wrapper)
    cpus = min([task, cpus])
    pool = multiprocessing.Pool(processes=cpus)
    n = 0
    for _ in pool.imap_unordered(nxbagel, variables_wrapper):
        n += 1
        sys.stdout.write('CPU: %3d generating trajectory: %d/%d\r' % (cpus, n, task))
    pool.close()

    with open('./runall.sh', 'w') as out:
        out.write(runall)

    os.system("chmod 777 runall.sh")
    print('\n\n Done\n')


def nxbagel_batch(inputs, j, start, end, in_path, slcr, sljb, sltm, slpt, slmm, tontx, tobgl, lbbls, lblpk, lbslp,
                  lbbst, tomkl, tompi, shell):
    ## This function will be called by gen_nxbagel
    ## This function generates runset for NxBagel calculation

    ### Note, this line doesn't check if slcr and sljb are appropriate for parallelization, be careful!!!
    bagelpal = int(slcr / sljb)

    batch = """#!/bin/sh
## script for NX/BAGEL
#SBATCH --nodes=1
#SBATCH --ntasks=%d
#SBATCH --time=%s
#SBATCH --job-name=%s-%d
#SBATCH --partition=%s
#SBATCH --mem=%dmb
#SBATCH --output=%%j.o.slurm
#SBATCH --error=%%j.e.slurm
%s
export MKL_DEBUG_CPU_TYPE=5

export BAGEL_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BAGELPAL=%s

export NX=%s/bin
export BAGEL=%s/bin/BAGEL
export BAGEL_LIB=%s/lib
export BLAS_LIB=%s
export LAPACK_LIB=%s
export SCALAPACK_LIB=%s
export BOOST_LIB=%s/lib
source %s/setvars.sh
export MPI=%s
export LD_LIBRARY_PATH=$MPI/lib:$BAGEL_LIB:$BLAS_LIB:$LAPACK_LIB:$SCALAPACK_LIB:$BOOST_LIB:$LD_LIBRARY_PATH
export PATH=$MPI/bin:$PATH

echo $SLURM_JOB_NAME

for ((i=%d;i<=%d;i++))
do
  export WORKDIR=%s/%s-$i
  cd $WORKDIR
  $NX/moldyn.pl > moldyn.log &
  sleep 5
done
wait
""" % (
        slcr, sltm, inputs, j + 1, slpt, int(slmm * slcr * 1.1), shell, bagelpal, tontx, tobgl, tobgl, lbbls, lblpk,
        lbslp, lbbst, tomkl, tompi, start, end, in_path, inputs)

    return batch


def nxbagel(var):
    ## This function prepares NxBagel calculation
    ## It generates TRAJECTORIES/TRAJ#/JOB_NAD, geom, veloc, controd.dyn for NxBagel calculations
    ## This function generates a backup slurm batch file for each calculation

    inputname, inputpath, slcr, sljb, sltm, slpt, slmm, in_temp, in_xyz, in_velo, tontx, tobgl, lbbls, \
        lblpk, lbslp, lbbst, tomkl, tompi, shell = var
    ### Note, this line doesn't check if slcr and sljb are appropriate for parallelization, be careful!!!
    bagelpal = int(slcr / sljb)

    if not os.path.exists('%s/JOB_NAD' % inputpath):
        os.makedirs('%s/JOB_NAD' % inputpath)

    runscript = """#!/bin/sh
## script for NX/BAGEL
#SBATCH --nodes=1
#SBATCH --ntasks=%s
#SBATCH --time=%s
#SBATCH --job-name=%s
#SBATCH --partition=%s
#SBATCH --mem=%dmb
#SBATCH --output=%%j.o.slurm
#SBATCH --error=%%j.e.slurm
%s
export BAGEL_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BAGELPAL=$SLURM_NTASKS

export NX=%s/bin
export BAGEL=%s/bin/BAGEL
export BAGEL_LIB=%s/lib
export BLAS_LIB=%s
export LAPACK_LIB=%s
export SCALAPACK_LIB=%s
export BOOST_LIB=%s/lib
source %s/setvars.sh
export MPI=%s
export LD_LIBRARY_PATH=$MPI/lib:$BAGEL_LIB:$BLAS_LIB:$LAPACK_LIB:$SCALAPACK_LIB:$BOOST_LIB:$LD_LIBRARY_PATH
export PATH=$MPI/bin:$PATH

export WORKDIR=%s
cd $WORKDIR
$NX/moldyn.pl > moldyn.log
""" % (bagelpal, sltm, inputname, slpt, int(slmm * 1.1), shell, tontx, tobgl, tobgl, lbbls, lblpk, lbslp, lbbst, tomkl,
       tompi, inputpath)

    with open('%s/%s.sh' % (inputpath, inputname), 'w') as out:
        out.write(runscript)

    if in_temp['orb'] != '':
        shutil.copy2('%s' % (in_temp['orb']), '%s/%s' % (inputpath, in_temp['orb']))

    if in_temp['jiri'] != '':
        with open('%s/jiri.inp' % inputpath, 'w') as out:
            out.write(in_temp['jiri'])

    if in_temp['sh'] != '':
        with open('%s/sh.inp' % inputpath, 'w') as out:
            out.write(in_temp['sh'])

    if in_temp['therm'] != '':
        with open('%s/therm.inp' % inputpath, 'w') as out:
            out.write(in_temp['therm'])

    with open('%s/control.dyn' % inputpath, 'w') as out:
        out.write(in_temp['control'])

    with open('%s/JOB_NAD/bagelinput.basis.inp' % inputpath, 'w') as out:
        out.write(in_temp['basis'])

    with open('%s/JOB_NAD/bagelinput.part1.inp' % inputpath, 'w') as out:
        out.write(in_temp['part1'])

    with open('%s/JOB_NAD/bagelinput.part2.inp' % inputpath, 'w') as out:
        out.write(in_temp['part2'])

    with open('%s/JOB_NAD/bagelinput.part3.inp' % inputpath, 'w') as out:
        out.write(in_temp['part3'])

    with open('%s/geom' % inputpath, 'w') as out:
        out.write(in_xyz)

    with open('%s/veloc' % inputpath, 'w') as out:
        out.write(in_velo)

    os.system("chmod 777 %s/%s.sh" % (inputpath, inputname))


def Unpack(ensemble, prog):
    ## This function unpacks initial condition to xyz and velocity

    xyz = ''
    velo = ''
    if prog == 'molcas':
        natom = int(len(ensemble))
        xyz = '%d\n\n' % natom
        for i in ensemble:
            xyz += '%-5s%30.16f%30.16f%30.16f\n' % (i[0], float(i[1]), float(i[2]), float(i[3]))
            velo += '%30.16f%30.16f%30.16f\n' % (float(i[4]), float(i[5]), float(i[6]))
    elif prog == 'newton':
        for i in ensemble:
            xyz += '%-5s%6.1f%30.16f%30.16f%30.16f%30.16f\n' % (
                i[0], float(i[8]), float(i[1]) * 1.88973, float(i[2]) * 1.88973, float(i[3]) * 1.88973, float(i[7]))
            velo += '%30.16f%30.16f%30.16f\n' % (float(i[4]), float(i[5]), float(i[6]))
    elif prog == 'bagel':
        xyz = []
        velo = []
        for i in ensemble:
            xyz.append([i[0], float(i[1]), float(i[2]), float(i[3])])
            velo.append([float(i[4]), float(i[5]), float(i[6])])
    elif prog == 'lammps':
        xyz = []
        velo = []
        for i in ensemble:
            xyz.append([str(i[1]), str(i[2]), str(i[3])])

    return xyz, velo


def gen_pyrai2md(cpus, ensemble, inputs, slpt, sltm, slmm, slnd, slcr, sljb, slin, qm, iformat, shell):
    ## This function will group PyRAI2MD calculations to individual runset
    ## this function will call pyrai2md_batch and pyrai2md to prepare files

    in_temp = open('input', 'r').read()
    in_path = os.getcwd()
    variables_wrapper = []
    runall = ''
    for j in range(slnd):
        runall += 'sbatch runset-%d.sh\n' % (j + 1)
        start = slin + j * sljb
        end = start + sljb - 1
        for i in range(sljb):
            if iformat != 'xz':
                # unpack initial condition to xyz and velocity
                in_xyz, in_velo = Unpack(ensemble[i + j * sljb], 'molcas')
            else:
                in_xyz, in_velo = UnpackXZ(ensemble[i + j * sljb])
            inputname = '%s-%s' % (inputs, i + start)
            inputpath = '%s/%s' % (in_path, inputname)
            # prepare calculations
            variables_wrapper.append([
                inputs, inputname, inputpath, slcr, sljb, sltm, slpt, slmm, in_temp, in_xyz, in_velo, qm, shell
            ])

        batch = pyrai2md_batch(inputs, j, start, end, in_path, slcr, sltm, slpt, slmm, shell)

        with open('./runset-%d.sh' % (j + 1), 'w') as out:
            out.write(batch)

        os.system("chmod 777 runset-%d.sh" % (j + 1))

    task = len(variables_wrapper)
    cpus = min([task, cpus])
    pool = multiprocessing.Pool(processes=cpus)
    n = 0
    for _ in pool.imap_unordered(pyrai2md, variables_wrapper):
        n += 1
        sys.stdout.write('CPU: %3d generating trajectory: %d/%d\r' % (cpus, n, task))
    pool.close()

    with open('./runall.sh', 'w') as out:
        out.write(runall)
    os.system("chmod 777 runall.sh")

    print('\n\n Done\n')


def pyrai2md_batch(inputs, j, start, end, in_path, slcr, sltm, slpt, slmm, shell):
    ## This function will be called by gen_pyrai2md
    ## This function generates runset for PyRAI2MD calculation

    batch = """#!/bin/sh
## script for PyRAI2MD
#SBATCH --nodes=1
#SBATCH --cpus-per-task=%d
#SBATCH --time=%s
#SBATCH --job-name=%s-%d
#SBATCH --partition=%s
#SBATCH --mem=%dmb
#SBATCH --output=%%j.o.slurm
#SBATCH --error=%%j.e.slurm
%s
export INPUT=input

for ((i=%d;i<=%d;i++))
do
  export WORKDIR=%s/%s-$i
  cd $WORKDIR
  pyrai2md $INPUT &
  sleep 5
done
wait
""" % (slcr, sltm, inputs, j + 1, slpt, int(slmm * slcr * 1.1), shell, start, end, in_path, inputs)

    return batch


def pyrai2md(var):
    ## This function prepares PyRAI2MD calculation
    ## This function generates a backup slurm batch file for each calculation

    inputs, inputname, inputpath, slcr, sljb, sltm, slpt, slmm, in_temp, in_xyz, in_velo, qm, shell = var
    ### Note, this line doesn't check if slcr and sljb are appropriate for parallelization, be careful!!!
    ncpu = int(
        slcr / sljb)

    if not os.path.exists('%s' % inputpath):
        os.makedirs('%s' % inputpath)

    in_temp = update_pyrai2md_input(in_temp, inputname)

    # copy molcas files
    if os.path.exists('%s.molcas' % inputs) and not os.path.exists('%s/%s.molcas' % (inputpath, inputs)):
        shutil.copy2('%s.molcas' % inputs, '%s/%s.molcas' % (inputpath, inputname))

    if os.path.exists('%s.StrOrb' % inputs) and not os.path.exists('%s/%s.StrOrb' % (inputpath, inputs)):
        shutil.copy2('%s.StrOrb' % inputs, '%s/%s.StrOrb' % (inputpath, inputname))

    if os.path.exists('%s.JobIph' % inputs) and not os.path.exists('%s/%s.JobIph' % (inputpath, inputs)):
        shutil.copy2('%s.JobIph' % inputs, '%s/%s.JobIph' % (inputpath, inputname))

    # copy bagel files
    if os.path.exists('%s.bagel' % inputs) and not os.path.exists('%s/%s.bagel' % (inputpath, inputs)):
        shutil.copy2('%s.bagel' % inputs, '%s/%s.bagel' % (inputpath, inputname))

    if os.path.exists('%s.archive' % inputs) and not os.path.exists('%s/%s.archive' % (inputpath, inputs)):
        shutil.copy2('%s.archive' % inputs, '%s/%s.archive' % (inputpath, inputname))

    # copy orca files
    if os.path.exists('%s.orca' % inputs) and not os.path.exists('%s/%s.orca' % (inputpath, inputs)):
        shutil.copy2('%s.orca' % inputs, '%s/%s.orca' % (inputpath, inputname))

    # copy oqp files
    if os.path.exists('%s.openqp' % inputs) and not os.path.exists('%s/%s.openqp' % (inputpath, inputs)):
        shutil.copy2('%s.openqp' % inputs, '%s/%s.openqp' % (inputpath, inputname))

    if os.path.exists('%s.json' % inputs) and not os.path.exists('%s/guess.json' % inputpath):
        shutil.copy2('%s.json' % inputs, '%s/guess.json' % inputpath)

    # copy xtb files
    if os.path.exists('%s.xtb' % inputs) and not os.path.exists('%s/%s.xtb' % (inputpath, inputs)):
        shutil.copy2('%s.xtb' % inputs, '%s/%s.xtb' % (inputpath, inputname))

    # copy charge file
    if os.path.exists('%s.charge' % inputs) and not os.path.exists('%s/%s.charge' % (inputpath, inputs)):
        shutil.copy2('%s.charge' % inputs, '%s/%s.charge' % (inputpath, inputname))

    # copy basis set file
    if os.path.exists('%s.basis' % inputs) and not os.path.exists('%s/%s.basis' % (inputpath, inputs)):
        shutil.copy2('%s.basis' % inputs, '%s/%s.basis' % (inputpath, inputname))

    runscript = """#!/bin/sh
## script for PyRAI2MD
#SBATCH --nodes=1
#SBATCH --cpus-per-task=%s
#SBATCH --time=%s
#SBATCH --job-name=%s
#SBATCH --partition=%s
#SBATCH --mem=%dmb
#SBATCH --output=%%j.o.slurm
#SBATCH --error=%%j.e.slurm
%s
export INPUT=input
export WORKDIR=%s

cd $WORKDIR
pyrai2md $INPUT

""" % (ncpu, sltm, inputname, slpt, int(slmm * 1.1), shell, inputpath)

    with open('%s/run_PyRAI2MD.sh' % inputpath, 'w') as out:
        out.write(runscript)

    with open('%s/input' % inputpath, 'w') as out:
        out.write(in_temp)

    with open('%s/%s.xyz' % (inputpath, inputname), 'w') as out:
        out.write(in_xyz)

    with open('%s/%s.velo' % (inputpath, inputname), 'w') as out:
        out.write(in_velo)

    os.system("chmod 777 %s/run_PyRAI2MD.sh" % inputpath)


def update_pyrai2md_input(in_temp, inputname):
    ## first edit input - change the title
    ## second copy neural network
    inputs = ''
    for line in in_temp.splitlines():
        if 'title' in line:
            inputs += 'title %s\n' % inputname
        else:
            inputs += '%s\n' % line

    return inputs


def UnpackXZ(ensemble):
    xyz = ensemble['txyz']
    velo = ensemble['velo']

    xyz = '\n'.join(xyz) + '\n'
    velo = '\n'.join(velo) + '\n'

    return xyz, velo


def gen_fromage(cpus, ensemble, inputs, slpt, sltm, slmm, slnd, slcr, sljb, slin, tomlcs, toxtb, iformat, shell):
    ## This function will group fromage calculations to individual runset
    ## this function will call fromage_batch and molcas to prepare files

    in_path = os.getcwd()

    runall = ''
    variables_wrapper = []
    for j in range(slnd):
        runall += 'sbatch runset-%d.sh\n' % (j + 1)
        start = slin + j * sljb
        end = start + sljb - 1
        for i in range(sljb):
            if iformat != 'xz':
                in_xyz, in_velo = Unpack(ensemble[i + j * sljb],
                                         'molcas')  # unpack initial condition to xyz and velocity
            else:
                in_xyz, in_velo = UnpackXZ(ensemble[i + j * sljb])
            inputname = '%s-%s' % (inputs, i + start)
            inputpath = '%s/%s' % (in_path, inputname)
            variables_wrapper.append([
                inputs, inputname, inputpath, slmm, in_xyz, in_velo, tomlcs, toxtb, shell
            ])  # prepare calculations

        batch = fromage_batch(inputs, j, start, end, in_path, slcr, sltm, slpt, slmm, tomlcs, toxtb, shell)

        with open('./runset-%d.sh' % (j + 1), 'w') as out:
            out.write(batch)

        os.system("chmod 777 runset-%d.sh" % (j + 1))

    task = len(variables_wrapper)
    cpus = min([task, cpus])
    pool = multiprocessing.Pool(processes=cpus)
    n = 0
    for _ in pool.imap_unordered(fromage, variables_wrapper):
        n += 1
        sys.stdout.write('CPU: %3d generating trajectory: %d/%d\r' % (cpus, n, task))
    pool.close()

    with open('./runall.sh', 'w') as out:
        out.write(runall)

    os.system("chmod 777 runall.sh")

    print('\n\n Done\n')


def fromage_batch(inputs, j, start, end, in_path, slcr, sltm, slpt, slmm, tomlcs, toxtb, shell):
    ## This function will be called by gen_fromage function
    ## This function generates runset for fromage calculation

    batch = """#!/bin/sh
## script for OpenMalCas
#SBATCH --nodes=1
#SBATCH --ntasks=%d
#SBATCH --time=%s
#SBATCH --job-name=%s-%d
#SBATCH --partition=%s
#SBATCH --mem=%dmb
#SBATCH --output=%%j.o.slurm
#SBATCH --error=%%j.e.slurm
%s
export MOLCAS_NPROCS=1
export MOLCAS_MEM=%d
export MOLCAS_PRINT=2
export OMP_NUM_THREADS=1
export MOLCAS=%s
export PATH=$MOLCAS/bin:$PATH

export PATH=$PATH:/%s

echo $SLURM_JOB_NAME

if [ -d "/srv/tmp" ]
then
 export LOCAL_TMP=/srv/tmp
else
 export LOCAL_TMP=/tmp
fi

for ((i=%d;i<=%d;i++))
do
  export INPUT="%s-$i"
  export WORKDIR="%s/%s-$i"
  export MOLCAS_PROJECT=$INPUT
  export MOLCAS_WORKDIR=$LOCAL_TMP/$USER/$SLURM_JOB_ID
  cd $WORKDIR
  fro_run.py &
  sleep 5
done
wait
rm -r $MOLCAS_WORKDIR

""" % (slcr, sltm, inputs, j + 1, slpt, int(slmm * slcr * 1.333), shell, slmm, tomlcs, toxtb, start, end,
       inputs, in_path, inputs)

    return batch


def fromage(var):
    ## This function prepares fromage calculation
    ## This function copy fromage.in shell.xyz and generates mol.init.xyz
    ## This function creates mh (mh.temp .StrOrb), ml (ml.temp xtb_charges.pc, xtb.input), rl (rl.temp) folders
    ## This function generates a backup slurm batch file for each calculation
    inputs, inputname, inputpath, slmm, in_xyz, in_velo, tomlcs, toxtb, shell = var

    if not os.path.exists('%s' % inputpath):
        os.makedirs('%s' % inputpath)

    runscript = """#!/bin/sh
## backup script for OpenMalCas
## $INPUT and $WORKDIR do not belong to OpenMolCas
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=23:50:00
#SBATCH --job-name=%s
#SBATCH --partition=short
#SBATCH --mem=%dmb
#SBATCH --output=%%j.o.slurm
#SBATCH --error=%%j.e.slurm
%s
if [ -d "/srv/tmp" ]
then
 export LOCAL_TMP=/srv/tmp
else
 export LOCAL_TMP=/tmp
fi

export INPUT=%s
export WORKDIR=%s
export MOLCAS_NPROCS=1
export MOLCAS_MEM=%d
export MOLCAS_PRINT=2
export MOLCAS_PROJECT=$INPUT
export OMP_NUM_THREADS=1
export MOLCAS=%s
export MOLCAS_WORKDIR=$LOCAL_TMP/$USER/$SLURM_JOB_ID
export PATH=$MOLCAS/bin:$PATH

export PATH=$PATH:/%s

cd $WORKDIR
fro_run.py
rm -r $MOLCAS_WORKDIR
""" % (inputname, int(slmm * 1.333), shell, inputname, inputpath, slmm, tomlcs, toxtb)

    with open('%s/%s.sh' % (inputpath, inputname), 'w') as out:
        out.write(runscript)

    os.system("chmod 777 %s/%s.sh" % (inputpath, inputname))

    with open('%s/mol.init.xyz' % inputpath, 'w') as out:
        out.write(in_xyz)

    with open('%s/velocity' % inputpath, 'w') as out:
        out.write(in_velo)

    shutil.copy2('fromage.in', '%s/fromage.in' % inputpath)
    shutil.copy2('shell.xyz', '%s/shell.xyz' % inputpath)

    if not os.path.exists('%s/mh' % inputpath):
        os.makedirs('%s/mh' % inputpath)

    shutil.copy2('mh.temp', '%s/mh/mh.temp' % inputpath)
    shutil.copy2('%s.StrOrb' % inputs, '%s/mh/%s.StrOrb' % (inputpath, inputname))

    if not os.path.exists('%s/ml' % inputpath):
        os.makedirs('%s/ml' % inputpath)

    shutil.copy2('ml.temp', '%s/ml/ml.temp' % inputpath)
    shutil.copy2('xtb.input', '%s/ml/xtb.input' % inputpath)
    shutil.copy2('xtb_charge.pc', '%s/ml/xtb_charge.pc' % inputpath)

    if not os.path.exists('%s/rl' % inputpath):
        os.makedirs('%s/rl' % inputpath)

    shutil.copy2('rl.temp', '%s/rl/rl.temp' % inputpath)


def gen_orca(cpus, ensemble, inputs, slpt, sltm, slmm, slnd, slcr, sljb, slin, toorca, iformat, shell):
    ## This function will group ORCA calculations to individual runset
    ## this function will call orca_batch and orca to prepare files

    in_temp = open('%s.orca' % inputs, 'r').read()
    in_path = os.getcwd()

    runall = ''
    variables_wrapper = []
    for j in range(slnd):
        runall += 'sbatch runset-%d.sh\n' % (j + 1)
        start = slin + j * sljb
        end = start + sljb - 1
        for i in range(sljb):
            if iformat != 'xz':
                # unpack initial condition to xyz and velocity
                in_xyz, in_velo = Unpack(ensemble[i + j * sljb], 'molcas')
            else:
                in_xyz, in_velo = UnpackXZ(ensemble[i + j * sljb])
            inputname = '%s-%s' % (inputs, i + start)
            inputpath = '%s/%s' % (in_path, inputname)
            # prepare calculations
            variables_wrapper.append([inputname, inputpath, slcr, sljb, slmm, in_temp, in_xyz, in_velo, toorca, shell])

        batch = orca_batch(inputs, j, start, end, in_path, slcr, sltm, slpt, slmm, toorca, shell)

        with open('./runset-%d.sh' % (j + 1), 'w') as run:
            run.write(batch)

    task = len(variables_wrapper)
    cpus = min([task, cpus])
    pool = multiprocessing.Pool(processes=cpus)
    n = 0
    for _ in pool.imap_unordered(orca, variables_wrapper):
        n += 1
        sys.stdout.write('CPU: %3d generating trajectory: %d/%d\r' % (cpus, n, task))
    pool.close()

    with open('./runall.sh', 'w') as out:
        out.write(runall)

    os.system("chmod 777 runall.sh")

    print('\n\n Done\n')


def orca_batch(inputs, j, start, end, in_path, slcr, sltm, slpt, slmm, toorca, shell):
    ## This function will be called by gen_orca function
    ## This function generates runset for ORCA calculation

    batch = """#!/bin/sh
## script for ORCA
#SBATCH --nodes=1
#SBATCH --ntasks=%d
#SBATCH --time=%s
#SBATCH --job-name=%s-%d
#SBATCH --partition=%s
#SBATCH --mem=%dmb
#SBATCH --output=%%j.o.slurm
#SBATCH --error=%%j.e.slurm
%s
export ORCA_EXE=%s

echo $SLURM_JOB_NAME

for ((i=%d;i<=%d;i++))
do
  export INPUT="%s-$i"
  export WORKDIR="%s/%s-$i"
  cd $WORKDIR
  $ORCA_EXE/orca $INPUT.inp > $INPUT.out &
  sleep 5
done
wait

""" % (
        slcr, sltm, inputs, j + 1, slpt, int(slmm * slcr * 1.0), shell, toorca,
        start, end, inputs, in_path, inputs)

    return batch


def orca(var):
    ## This function prepares ORCA calculation
    ## It generates .inp .xyz .velocity.xyz
    ## This function generates a backup slurm batch file for each calculation

    inputname, inputpath, slcr, sljb, slmm, in_temp, in_xyz, in_velo, toorca, shell = var
    orcapal = int(slcr / sljb)

    if not os.path.exists('%s' % inputpath):
        os.makedirs('%s' % inputpath)

    runscript = """#!/bin/sh
## backup script for ORCA
#SBATCH --nodes=1
#SBATCH --ntasks=%s
#SBATCH --time=23:50:00
#SBATCH --job-name=%s
#SBATCH --partition=normal
#SBATCH --mem=%dmb
#SBATCH --output=%%j.o.slurm
#SBATCH --error=%%j.e.slurm
%s
export INPUT=%s
export WORKDIR=%s

export ORCA_EXE=%s

cd $WORKDIR
$ORCA_EXE/orca $INPUT.inp > $INPUT.out

""" % (orcapal, inputname, int(slmm * orcapal * 1.0), shell, inputname, inputpath, toorca)

    in_xyz = '\n'.join(in_xyz.splitlines()[2:])
    in_temp += in_xyz + '\n*\n'

    with open('%s/%s.inp' % (inputpath, inputname), 'w') as out:
        out.write(in_temp)

    with open('%s/%s.sh' % (inputpath, inputname), 'w') as out:
        out.write(runscript)

    with open('%s/%s.velocity.xyz' % (inputpath, inputname), 'w') as out:
        out.write(in_velo)

    os.system("chmod 777 %s/%s.sh" % (inputpath, inputname))


def gen_oqp(cpus, ensemble, inputs, slpt, sltm, slmm, slnd, slcr, sljb, slin, tooqp, iformat, shell):
    ## This function will group OQP calculations to individual runset
    ## this function will call oqp_batch and oqp to prepare files

    in_temp = open('%s.openqp' % inputs, 'r').read()
    in_path = os.getcwd()

    runall = ''
    variables_wrapper = []
    for j in range(slnd):
        runall += 'sbatch runset-%d.sh\n' % (j + 1)
        start = slin + j * sljb
        end = start + sljb - 1
        for i in range(sljb):
            if iformat != 'xz':
                # unpack initial condition to xyz and velocity
                in_xyz, in_velo = Unpack(ensemble[i + j * sljb], 'molcas')
            else:
                in_xyz, in_velo = UnpackXZ(ensemble[i + j * sljb])
            inputname = '%s-%s' % (inputs, i + start)
            inputpath = '%s/%s' % (in_path, inputname)
            # prepare calculations
            variables_wrapper.append([
                inputs, inputname, inputpath, slcr, sljb, slmm, in_temp, in_xyz, in_velo, tooqp, shell
            ])

        batch = oqp_batch(inputs, j, start, end, in_path, slcr, sljb, sltm, slpt, slmm, tooqp, shell)

        with open('./runset-%d.sh' % (j + 1), 'w') as run:
            run.write(batch)

    task = len(variables_wrapper)
    cpus = min([task, cpus])
    pool = multiprocessing.Pool(processes=cpus)
    n = 0
    for _ in pool.imap_unordered(oqp, variables_wrapper):
        n += 1
        sys.stdout.write('CPU: %3d generating trajectory: %d/%d\r' % (cpus, n, task))
    pool.close()

    with open('./runall.sh', 'w') as out:
        out.write(runall)

    os.system("chmod 777 runall.sh")

    print('\n\n Done\n')


def oqp_batch(inputs, j, start, end, in_path, slcr, sljb, sltm, slpt, slmm, tooqp, shell):
    ## This function will be called by gen_oqp function
    ## This function generates runset for OQP calculation
    oqppal = int(slcr / sljb)

    batch = """#!/bin/sh
## script for OpenQP
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=%s
#SBATCH --cpus-per-task=1
#SBATCH --time=%s
#SBATCH --job-name=%s-%d
#SBATCH --partition=%s
#SBATCH --mem=%dmb
#SBATCH --output=%%j.o.slurm
#SBATCH --error=%%j.e.slurm
%s
export OMP_NUM_THREADS=1
export OPENQP_ROOT=%s

echo $SLURM_JOB_NAME

for ((i=%d;i<=%d;i++))
do
  export INPUT="%s-$i"
  export WORKDIR="%s/%s-$i"
  cd $WORKDIR
  mpirun -np $SLURM_NTASKS openqp $INPUT.inp &
  sleep 5
done
wait

""" % (
        slcr, sltm, inputs, j + 1, slpt, int(slmm * slcr * 1.0), shell, tooqp,
        start, end, inputs, in_path, inputs)

    return batch


def oqp(var):
    ## This function prepares OQP calculation
    ## It generates .inp .xyz .velocity.xyz
    ## This function generates a backup slurm batch file for each calculation

    inputs, inputname, inputpath, slcr, sljb, slmm, in_temp, in_xyz, in_velo, tooqp, shell = var
    oqppal = int(slcr / sljb)

    if not os.path.exists('%s' % inputpath):
        os.makedirs('%s' % inputpath)

    # copy guess files
    if os.path.exists('%s.json' % inputs) and not os.path.exists('%s/guess.json' % inputpath):
        shutil.copy2('%s.json' % inputs, '%s/guess.json' % inputpath)

    runscript = """#!/bin/sh
## backup script for OpenQP
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=%s
#SBATCH --cpus-per-task=1
#SBATCH --time=23:50:00
#SBATCH --job-name=%s
#SBATCH --partition=normal
#SBATCH --mem=%dmb
#SBATCH --output=%%j.o.slurm
#SBATCH --error=%%j.e.slurm
%s
export INPUT=%s
export WORKDIR=%s

export OMP_NUM_THREADS=1
export OPENQP_ROOT=%s


cd $WORKDIR
mpirun -np $SLURM_NTASKS openqp $INPUT.inp

""" % (oqppal, inputname, int(slmm * oqppal * 1.0), shell, inputname, inputpath, tooqp)

    new_in_temp = ''
    for line in in_temp.split():
        if 'system=' in line:
            new_in_temp += 'system=%s.xyz\n' % inputname
        elif 'type=json' in line:
            new_in_temp += 'type=json\nfile=guess.json\n'
        elif 'file=' in line:
            continue
        else:
            new_in_temp += '%s\n' % line

    with open('%s/%s.inp' % (inputpath, inputname), 'w') as out:
        out.write(new_in_temp)

    with open('%s/%s.xyz' % (inputpath, inputname), 'w') as out:
        out.write(in_xyz)

    with open('%s/%s.sh' % (inputpath, inputname), 'w') as out:
        out.write(runscript)

    with open('%s/%s.velocity.xyz' % (inputpath, inputname), 'w') as out:
        out.write(in_velo)

    os.system("chmod 777 %s/%s.sh" % (inputpath, inputname))


def gen_lmp(cpus, ensemble, inputs, slpt, sltm, slmm, slnd, slcr, sljb, slin, tolmp, iformat, shell):
    ## This function will group lammps calculations to individual runset
    ## this function will call oqp_batch and oqp to prepare files

    in_temp = open('%s.in' % inputs, 'r').read()
    in_data = open('%s.data' % inputs, 'r').read()
    in_init = open('%s.in.init' % inputs, 'r').read()
    in_settings = open('%s.in.settings' % inputs, 'r').read()

    in_path = os.getcwd()

    runall = ''
    variables_wrapper = []
    for j in range(slnd):
        runall += 'sbatch runset-%d.sh\n' % (j + 1)
        start = slin + j * sljb
        end = start + sljb - 1
        for i in range(sljb):
            if iformat != 'xz':
                # unpack initial condition to xyz and velocity
                in_xyz, in_velo = Unpack(ensemble[i + j * sljb], 'lammps')
            else:
                in_xyz, in_velo = UnpackXZ(ensemble[i + j * sljb])
            inputname = '%s-%s' % (inputs, i + start)
            inputpath = '%s/%s' % (in_path, inputname)
            # prepare calculations
            variables_wrapper.append([
                inputs, inputname, inputpath, slcr, sljb, slmm, in_temp, in_data, in_init, in_settings,
                in_xyz, in_velo, tolmp, shell
            ])

        batch = lmp_batch(inputs, j, start, end, in_path, slcr, sljb, sltm, slpt, slmm, tolmp, shell)

        with open('./runset-%d.sh' % (j + 1), 'w') as run:
            run.write(batch)

    task = len(variables_wrapper)
    cpus = min([task, cpus])
    pool = multiprocessing.Pool(processes=cpus)
    n = 0
    for _ in pool.imap_unordered(lmp, variables_wrapper):
        n += 1
        sys.stdout.write('CPU: %3d generating trajectory: %d/%d\r' % (cpus, n, task))
    pool.close()

    with open('./runall.sh', 'w') as out:
        out.write(runall)

    os.system("chmod 777 runall.sh")

    print('\n\n Done\n')


def lmp_batch(inputs, j, start, end, in_path, slcr, sljb, sltm, slpt, slmm, tolmp, shell):
    ## This function will be called by gen_lmp function
    ## This function generates runset for lammps calculation
    oqppal = int(slcr / sljb)

    batch = """#!/bin/sh
## script for lammps
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=%s
#SBATCH --cpus-per-task=1
#SBATCH --time=%s
#SBATCH --job-name=%s-%d
#SBATCH --partition=%s
#SBATCH --mem=%dmb
#SBATCH --output=%%j.o.slurm
#SBATCH --error=%%j.e.slurm
%s
export OMP_NUM_THREADS=1
export LAMMPS=%s

echo $SLURM_JOB_NAME

for ((i=%d;i<=%d;i++))
do
  export INPUT="%s-$i"
  export WORKDIR="%s/%s-$i"
  cd $WORKDIR
  mpirun -np $SLURM_NTASKS $LAMMPS/bin/lmp -in $INPUT.in > $INPUT.log &
  sleep 5
done
wait

""" % (
        slcr, sltm, inputs, j + 1, slpt, int(slmm * slcr * 1.0), shell, tolmp,
        start, end, inputs, in_path, inputs)

    return batch


def lmp(var):
    ## This function prepares lammps calculation
    ## It generates .inp .xyz .velocity.xyz
    ## This function generates a backup slurm batch file for each calculation

    inputs, inputname, inputpath, slcr, sljb, slmm, in_temp, in_data, in_init, in_settings, in_xyz, in_velo, tolmp, shell = var
    oqppal = int(slcr / sljb)

    if not os.path.exists('%s' % inputpath):
        os.makedirs('%s' % inputpath)

    runscript = """#!/bin/sh
## backup script for lammps
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=%s
#SBATCH --cpus-per-task=1
#SBATCH --time=23:50:00
#SBATCH --job-name=%s
#SBATCH --partition=normal
#SBATCH --mem=%dmb
#SBATCH --output=%%j.o.slurm
#SBATCH --error=%%j.e.slurm
%s
export INPUT=%s
export WORKDIR=%s

export OMP_NUM_THREADS=1
export LAMMPS=%s


cd $WORKDIR
mpirun -np $SLURM_NTASKS $LAMMPS/bin/lmp -in $INPUT.in > $INPUT.log

""" % (oqppal, inputname, int(slmm * oqppal * 1.0), shell, inputname, inputpath, tolmp)
    in_temp = in_temp.splitlines()
    new_in_temp = ' include "%s.in.init"\n read_data "%s.data"\n include "%s.in.settings"\n\n' % (
        inputname, inputname, inputname
    )
    for line in in_temp:
        if 'include ' not in line and 'read_data ' not in line:
            new_in_temp += '%s\n' % line

    data = in_data.splitlines()
    natom = 0
    new_data = []
    for n, line in enumerate(data):
        if 'atoms' in line:
            natom = int(line.split()[0])
        if 'Atoms' in line:
            coord = data[n + 2: n + 2 + natom]
            temp = np.array([x.split() for x in coord])
            new = np.concatenate((temp[:, 0:4], in_xyz), axis=1).tolist()
            new_data = copy.deepcopy(data)
            new_data[n + 2: n + 2 + natom] = [' '.join(x) for x in new]
            break

    new_data = '\n'.join(new_data) + '\n'

    with open('%s/%s.in' % (inputpath, inputname), 'w') as out:
        out.write(new_in_temp)

    with open('%s/%s.data' % (inputpath, inputname), 'w') as out:
        out.write(new_data)

    with open('%s/%s.in.init' % (inputpath, inputname), 'w') as out:
        out.write(in_init)

    with open('%s/%s.in.settings' % (inputpath, inputname), 'w') as out:
        out.write(in_settings)

    with open('%s/%s.sh' % (inputpath, inputname), 'w') as out:
        out.write(runscript)

    os.system("chmod 777 %s/%s.sh" % (inputpath, inputname))


def read_xyzlist(inputs, nesmb):
    ensemble = []

    with open('%s.xyz' % inputs, 'r') as infile:
        file = infile.read().splitlines()

    natom = int(file[0])
    nmol = int(len(file) / (natom + 2))

    if nmol < nesmb:
        sys.exit('Not enough  structures!!! %s provided < %s requested.' % (nmol, nesmb))
    elif nmol > nesmb:
        print('More initial structures received. Skip %s - %s. ' % (nesmb + 1, nmol))

    file = file[0: nmol * (natom + 2)]

    mol = []
    for n, line in enumerate(file):
        if 0 < (n + 1) % (natom + 2) < 3:
            continue

        a, x, y, z = line.split()[0:4]

        mol.append([a, float(x), float(y), float(z), 0, 0, 0, 0, 0])

        if (n + 1) % (natom + 2) == 0:
            ensemble.append(copy.deepcopy(mol))
            mol = []

    return ensemble


if __name__ == '__main__':
    main(sys.argv)
